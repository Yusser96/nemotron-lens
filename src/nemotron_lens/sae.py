"""JumpReLU SAE inference module, on-disk-compatible with SAELens.

The storage format is exactly SAELens's native one — one folder per SAE holding

- ``cfg.json``               (config + free-form ``metadata``)
- ``sae_weights.safetensors`` (``W_enc (d_in, d_sae)``, ``W_dec (d_sae, d_in)``,
  ``b_enc``, ``b_dec``, ``threshold``)

so any SAE saved by :meth:`SAE.save_pretrained` loads with **stock sae_lens**::

    from sae_lens import SAE
    sae = SAE.from_pretrained("Yusser/nemotron-3-nano-30b-a3b-saes", "L2_resid_post/w16384_l0_10")

and vice versa — anything on the HuggingFace Hub in sae_lens format loads here
*without* sae_lens (or TransformerLens) installed::

    from nemotron_lens import SAE
    sae = SAE.from_pretrained("Yusser/nemotron-3-nano-30b-a3b-saes", "L2_resid_post/w16384_l0_10")

Forward convention (matching ``sae_lens.JumpReLUSAE`` with
``apply_b_dec_to_input=False`` / ``normalize_activations="none"``)::

    pre  = x @ W_enc + b_enc          # (optionally x - b_dec first)
    f    = relu(pre) * (pre > threshold)
    xhat = f @ W_dec + b_dec
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

SAE_CFG_FILENAME = "cfg.json"
SAE_WEIGHTS_FILENAME = "sae_weights.safetensors"

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class SAEConfig:
    """Mirrors the sae_lens v6 ``cfg.json`` schema (JumpReLU fields)."""

    d_in: int
    d_sae: int
    architecture: str = "jumprelu"
    dtype: str = "float32"
    device: str = "cpu"
    apply_b_dec_to_input: bool = False
    normalize_activations: str = "none"
    reshape_activations: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SAEConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        cfg = cls(**kwargs)
        if cfg.architecture != "jumprelu":
            raise NotImplementedError(
                f"nemotron_lens.SAE only implements architecture='jumprelu', got "
                f"{cfg.architecture!r}. Use sae_lens to load other architectures."
            )
        if cfg.normalize_activations != "none":
            raise NotImplementedError(
                f"normalize_activations={cfg.normalize_activations!r} is not supported; "
                "fold the normalization into the weights (nemotron-sae exports do this)."
            )
        return cfg


class SAE(nn.Module):
    """Standard JumpReLU SAE on raw activations (sae_lens-format load/save)."""

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in, d_sae = cfg.d_in, cfg.d_sae
        dt = _DTYPES[cfg.dtype]
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae, dtype=dt))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dt))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in, dtype=dt))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dt))
        self.threshold = nn.Parameter(torch.zeros(d_sae, dtype=dt))

    # ------------------------------------------------------------------ compute
    @property
    def d_in(self) -> int:
        return self.cfg.d_in

    @property
    def d_sae(self) -> int:
        return self.cfg.d_sae

    @property
    def metadata(self) -> dict[str, Any]:
        return self.cfg.metadata

    @property
    def hook_name(self) -> str | None:
        """HF module path this SAE was trained on (from metadata), if recorded."""
        return self.cfg.metadata.get("hook_name")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.W_enc.dtype)
        if self.cfg.apply_b_dec_to_input:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold).to(pre.dtype)

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return feats.to(self.W_dec.dtype) @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ---------------------------------------------------------------------- io
    def save_pretrained(self, path: str | Path) -> tuple[Path, Path]:
        """Write ``sae_weights.safetensors`` + ``cfg.json`` (sae_lens-native format)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        weights_path = path / SAE_WEIGHTS_FILENAME
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(weights_path))
        cfg_path = path / SAE_CFG_FILENAME
        cfg_path.write_text(json.dumps(self.cfg.to_dict(), indent=2))
        return weights_path, cfg_path

    # sae_lens calls this save_model; keep an alias so code written against
    # either library works.
    save_model = save_pretrained

    @classmethod
    def load_from_disk(cls, path: str | Path, device: str = "cpu") -> "SAE":
        path = Path(path)
        cfg_dict = json.loads((path / SAE_CFG_FILENAME).read_text())
        cfg = SAEConfig.from_dict(cfg_dict)
        cfg.device = device
        sae = cls(cfg)
        sd = load_file(str(path / SAE_WEIGHTS_FILENAME))
        missing = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"} - set(sd)
        if missing:
            raise KeyError(f"{path}: sae_weights.safetensors is missing keys {sorted(missing)}")
        sae.load_state_dict(sd)
        return sae.to(device)

    @classmethod
    def from_pretrained(cls, release: str, sae_id: str | None = None,
                        device: str = "cpu", revision: str = "main",
                        force_download: bool = False) -> "SAE":
        """Load from a local folder or any HuggingFace repo in sae_lens format.

        ``release`` is a local path (``sae_id`` ignored/None) or an HF repo id like
        ``"Yusser/nemotron-3-nano-30b-a3b-saes"`` with ``sae_id`` the folder inside
        it (``"L2_resid_post/w16384_l0_10"``) — the same call shape as
        ``sae_lens.SAE.from_pretrained``.
        """
        local = Path(release) if sae_id is None else Path(release) / sae_id
        if local.exists():
            return cls.load_from_disk(local, device=device)

        from huggingface_hub import hf_hub_download

        if sae_id is None:
            raise ValueError("Loading from the HF Hub requires sae_id (folder in the repo).")
        kwargs = dict(repo_id=release, revision=revision, force_download=force_download)
        cfg_file = hf_hub_download(filename=f"{sae_id}/{SAE_CFG_FILENAME}", **kwargs)
        hf_hub_download(filename=f"{sae_id}/{SAE_WEIGHTS_FILENAME}", **kwargs)
        return cls.load_from_disk(Path(cfg_file).parent, device=device)

    # ------------------------------------------------------------------ bridges
    def to_saelens(self, device: str | None = None):
        """Return this SAE as a stock ``sae_lens.JumpReLUSAE`` (requires sae-lens)."""
        from sae_lens import JumpReLUSAE, JumpReLUSAEConfig

        try:
            from sae_lens import SAEMetadata
        except ImportError:  # pragma: no cover - layout moved in some versions
            from sae_lens.saes.sae import SAEMetadata

        cfg = JumpReLUSAEConfig(
            d_in=self.cfg.d_in,
            d_sae=self.cfg.d_sae,
            dtype=self.cfg.dtype,
            device=device or self.cfg.device,
            apply_b_dec_to_input=self.cfg.apply_b_dec_to_input,
            normalize_activations=self.cfg.normalize_activations,
            metadata=SAEMetadata(**self.cfg.metadata),
        )
        sae = JumpReLUSAE(cfg)
        sae.load_state_dict(self.state_dict(), strict=False)
        return sae.to(device or self.cfg.device)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        md = self.cfg.metadata
        return (f"SAE(jumprelu, d_in={self.cfg.d_in}, d_sae={self.cfg.d_sae}, "
                f"hook={md.get('hook_name')!r}, model={md.get('model_name')!r})")


def list_saes(release: str, revision: str = "main") -> list[str]:
    """List the sae_ids available in an HF repo (folders containing cfg.json)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(release, revision=revision)
    return sorted(f[: -len("/" + SAE_CFG_FILENAME)]
                  for f in files if f.endswith(SAE_CFG_FILENAME) and "/" in f)
