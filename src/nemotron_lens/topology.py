"""Enumerate hookable activation sites in a Nemotron-family HF model.

Nemotron-H / Nemotron-3-Nano use one attribute name (``mixer``) for the block-level
module of every layer type (Mamba-2, attention, MoE, dense MLP), so sites are
classified by the concrete class of ``backbone.layers.{i}.mixer`` rather than by
name pattern. The result is a list of :class:`HookSite`, each with a canonical
TransformerLens-style name (``blocks.{L}.hook_{kind}``) and the real HF module path.

Residual-stream sites are block *I/O*, not modules; they are synthesized per layer
(``resid_pre`` = block input, ``resid_post`` = block output).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch.nn as nn

__all__ = ["HookSite", "enumerate_hooks", "dump_topology", "load_topology", "parse_hook_name"]

# Unqualified mixer class name -> component kind. Covers the Nemotron-H /
# Nemotron-3-Nano trust_remote_code modeling files.
_MIXER_CLASS_TO_KIND: dict[str, str] = {
    "NemotronHMamba2Mixer":     "mamba_out",
    "NemotronHMOE":             "moe_out",
    "NemotronHMLP":             "mlp_out",
    "NemotronHAttention":       "attn_out_prelinear",
    "NemotronHFlashAttention2": "attn_out_prelinear",
    "NemotronHSdpaAttention":   "attn_out_prelinear",
}

_MIXER_PATH_RE = re.compile(r"^(.*\.layers?\.(\d+))\.mixer$")

# Canonical hook-name grammar: blocks.{L}.hook_{kind}[.{idx}]
_CANONICAL_RE = re.compile(r"^blocks\.(\d+)\.hook_([a-z_]+?)(?:\.(\d+))?$")


@dataclass
class HookSite:
    """A single hookable point in the model."""

    component_kind: str          # resid_pre | resid_post | mamba_out | mlp_out |
                                 # attn_out_prelinear | moe_out | expert | shared_expert
    layer: int
    module_path: str             # dotted path from the model root ("<residual:L:pre|post>" for resid)
    extra: dict | None = field(default=None)

    @property
    def name(self) -> str:
        """Canonical TransformerLens-style hook name."""
        if self.component_kind == "expert" and self.extra is not None:
            return f"blocks.{self.layer}.hook_expert.{self.extra['expert_idx']}"
        return f"blocks.{self.layer}.hook_{self.component_kind}"

    @property
    def is_residual(self) -> bool:
        return self.component_kind in ("resid_pre", "resid_post")


def parse_hook_name(name: str) -> tuple[int, str] | None:
    """``"blocks.13.hook_moe_out"`` -> ``(13, "moe_out")``;
    ``"blocks.13.hook_expert.5"`` -> ``(13, "expert.5")``; None if not canonical."""
    m = _CANONICAL_RE.match(name)
    if m is None:
        return None
    layer, kind, idx = int(m.group(1)), m.group(2), m.group(3)
    return layer, (f"{kind}.{idx}" if idx is not None else kind)


def enumerate_hooks(model: nn.Module) -> list[HookSite]:
    """Walk every ``*.layers.{i}.mixer`` module and emit a HookSite per block kind,
    plus synthetic residual sites per layer."""
    sites: list[HookSite] = []
    n_layers = 0
    for name, mod in model.named_modules():
        m = _MIXER_PATH_RE.match(name)
        if m is None:
            continue
        layer = int(m.group(2))
        n_layers = max(n_layers, layer + 1)

        kind = _MIXER_CLASS_TO_KIND.get(type(mod).__name__)
        if kind is None:
            continue
        sites.append(HookSite(component_kind=kind, layer=layer, module_path=name))

        if kind == "moe_out":
            experts = getattr(mod, "experts", None)
            if experts is not None:
                for ename, _e in experts.named_children():  # "0", "1", ...
                    sites.append(HookSite(
                        component_kind="expert", layer=layer,
                        module_path=f"{name}.experts.{ename}",
                        extra={"expert_idx": int(ename)},
                    ))
            if hasattr(mod, "shared_experts"):
                sites.append(HookSite(
                    component_kind="shared_expert", layer=layer,
                    module_path=f"{name}.shared_experts",
                ))

    for L in range(n_layers):
        sites.append(HookSite(component_kind="resid_pre", layer=L, module_path=f"<residual:{L}:pre>"))
        sites.append(HookSite(component_kind="resid_post", layer=L, module_path=f"<residual:{L}:post>"))

    sites.sort(key=lambda s: (s.layer, s.component_kind, s.module_path))
    return sites


def dump_topology(sites: list[HookSite], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "n_sites": len(sites), "sites": [asdict(s) for s in sites]}
    out_path.write_text(json.dumps(payload, indent=2))


def load_topology(path: str | Path) -> list[HookSite]:
    payload = json.loads(Path(path).read_text())
    return [HookSite(**s) for s in payload["sites"]]
