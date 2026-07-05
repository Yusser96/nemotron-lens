"""TransformerLens-style hooked wrapper around Nemotron hybrid HF models.

``HookedNemotron`` wraps a plain ``AutoModelForCausalLM`` (Nemotron-3-Nano or any
Nemotron-H-family checkpoint) with:

- ``hook_points()``       — list every hookable location (canonical names)
- ``run_with_cache()``    — capture activations at named locations
- ``run_with_hooks()``    — observe or *edit* activations mid-forward
- ``run_with_saes()``     — splice SAE reconstructions into the forward pass
- ``stream_activations()``— yield flat (n_tokens, d) activation batches for SAE training

Hook names come in two interchangeable forms, accepted everywhere:

- canonical: ``blocks.{L}.hook_resid_post``, ``blocks.{L}.hook_mamba_out``,
  ``blocks.{L}.hook_moe_out``, ``blocks.{L}.hook_attn_out_prelinear``,
  ``blocks.{L}.hook_expert.{i}``, ...
- raw HF module paths: ``backbone.layers.13.mixer`` — this is what the published
  SAEs carry in ``metadata.hook_name``.

Residual-stream hooks are block I/O: ``hook_resid_pre`` is the block's input
hidden-states, ``hook_resid_post`` its output — identical to what the
nemotron-sae training pipeline captured.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
import torch.nn as nn

from nemotron_lens.topology import HookSite, dump_topology, enumerate_hooks, parse_hook_name

log = logging.getLogger(__name__)

__all__ = ["HookedNemotron", "EarlyExit"]

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# fn(acts, hook_name) -> replacement tensor or None (observe-only)
HookFn = Callable[[torch.Tensor, str], torch.Tensor | None]

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class EarlyExit(BaseException):
    """Raised inside a capture hook to short-circuit the forward pass once every
    requested activation has been captured. BaseException so model code that
    catches ``Exception`` cannot swallow it."""


@dataclass
class _Resolved:
    """A hook name resolved to a module + where to attach."""

    name: str            # the name as given by the caller
    module: nn.Module
    point: str           # "input" (forward_pre) | "output" (forward hook)
    module_path: str     # real HF module path (block path for residual hooks)


def _first_tensor(x: Any) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def _replace_first(output: Any, new: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    return new


class HookedNemotron:
    """A hooked Nemotron model. Build with :meth:`from_pretrained`, or wrap an
    already-loaded ``(model, tokenizer)`` pair directly."""

    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.sites: list[HookSite] = enumerate_hooks(model)
        self._site_by_name: dict[str, HookSite] = {s.name: s for s in self.sites}
        self._module_paths: dict[str, nn.Module] = dict(model.named_modules())
        if not self.sites:
            log.warning(
                "No hookable mixer sites found in %s — only raw module-path hooks "
                "will work. (Is this a Nemotron-H-family model?)", type(model).__name__,
            )

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        local_files_only: bool = False,
        **hf_kwargs: Any,
    ) -> "HookedNemotron":
        """Load model + tokenizer from HuggingFace. The model is frozen, in eval
        mode, with the KV cache disabled (we hook, we don't generate — set
        ``model.config.use_cache = True`` back if you need generation).

        Nemotron's Mamba-2 layers need ``mamba-ssm`` + ``causal-conv1d``
        (``pip install nemotron-lens[model]``) and CUDA.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, Any] = dict(
            torch_dtype=_DTYPES.get(dtype, dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            **hf_kwargs,
        )
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except (TypeError, ValueError):
            # Custom modeling code may reject attn_implementation; retry without.
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        model.config.use_cache = False
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, tokenizer)

    # ------------------------------------------------------------------- props
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def d_model(self) -> int | None:
        return getattr(self.model.config, "hidden_size", None)

    # ------------------------------------------------------------- hook points
    def hook_points(
        self,
        *,
        kinds: Sequence[str] | None = None,
        layers: Sequence[int] | None = None,
        include_experts: bool = False,
    ) -> list[str]:
        """Canonical names of all hookable locations.

        ``kinds`` filters by component kind (``resid_post``, ``mamba_out``,
        ``moe_out``, ``attn_out_prelinear``, ...); per-expert /
        shared-expert sites are hidden unless ``include_experts=True`` (128
        experts x 23 MoE layers is a lot of noise).
        """
        out = []
        for s in self.sites:
            if not include_experts and s.component_kind in ("expert", "shared_expert"):
                continue
            if kinds is not None and s.component_kind not in kinds:
                continue
            if layers is not None and s.layer not in layers:
                continue
            out.append(s.name)
        return out

    def hook_name(self, layer: int, component: str) -> str:
        """``(13, "moe_out")`` -> ``"blocks.13.hook_moe_out"`` (validated)."""
        name = f"blocks.{layer}.hook_{component}"
        self.resolve(name)
        return name

    def module_path(self, name: str) -> str:
        """The real HF module path behind a hook name (block path for residuals).
        This is the value stored in the published SAEs' ``metadata.hook_name``."""
        return self.resolve(name).module_path

    def save_topology(self, path) -> None:
        """Dump the enumerated sites to JSON (compatible with nemotron-sae's
        ``model_topology.json``)."""
        dump_topology(self.sites, path)

    def _block_module(self, layer: int) -> tuple[nn.Module, str]:
        for tmpl in (f"backbone.layers.{layer}", f"model.layers.{layer}",
                     f"layers.{layer}", f"transformer.h.{layer}"):
            mod = self._module_paths.get(tmpl)
            if mod is not None:
                return mod, tmpl
        raise KeyError(f"Could not locate the block module for layer {layer}.")

    def resolve(self, name: str) -> _Resolved:
        """Resolve a canonical hook name or raw HF module path."""
        parsed = parse_hook_name(name)
        if parsed is not None:
            layer, component = parsed
            if component == "resid_pre":
                mod, path = self._block_module(layer)
                return _Resolved(name, mod, "input", path)
            if component == "resid_post":
                mod, path = self._block_module(layer)
                return _Resolved(name, mod, "output", path)
            site = self._site_by_name.get(name)
            if site is None:
                have = sorted({s.component_kind for s in self.sites if s.layer == layer})
                raise KeyError(
                    f"No hook point {name!r}: layer {layer} has kinds {have}. "
                    f"(A Nemotron layer is Mamba OR attention OR MoE, not all three.)"
                )
            return _Resolved(name, self._module_paths[site.module_path], "output", site.module_path)

        mod = self._module_paths.get(name)
        if mod is not None:
            return _Resolved(name, mod, "output", name)
        raise KeyError(
            f"{name!r} is neither a canonical hook name (blocks.L.hook_*) nor a "
            f"module path of {type(self.model).__name__}."
        )

    # ---------------------------------------------------------------- tokenize
    def to_tokens(
        self, text: str | Sequence[str], *, max_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize to ``(input_ids, attention_mask)`` on the model device.

        Batches are **left-padded**: Mamba layers process the sequence as a causal
        recurrence, so right-aligned real tokens (pads first, masked out) keep the
        recurrent state and end-relative positions correct.
        """
        if self.tokenizer is None:
            raise RuntimeError("This HookedNemotron was built without a tokenizer.")
        tok = self.tokenizer
        prev = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(
                list(text) if not isinstance(text, str) else text,
                return_tensors="pt", padding=True,
                truncation=max_length is not None, max_length=max_length,
            )
        finally:
            tok.padding_side = prev
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _prepare(self, input: Any, max_length: int | None = None):
        if isinstance(input, torch.Tensor):
            return input.to(self.device), None
        if isinstance(input, tuple) and len(input) == 2:
            ids, mask = input
            return ids.to(self.device), (mask.to(self.device) if mask is not None else None)
        return self.to_tokens(input, max_length=max_length)

    # ----------------------------------------------------------------- hooking
    @contextmanager
    def hooks(self, fwd_hooks: Iterable[tuple[str, HookFn]]) -> Iterator[None]:
        """Register ``(hook_name, fn)`` pairs for the duration of the context.

        ``fn(acts, hook_name)`` receives the activation tensor — the block-input
        hidden states for ``hook_resid_pre``, else the module output's first
        tensor — and may return a replacement tensor (or None to observe only).
        """
        handles: list[torch.utils.hooks.RemovableHandle] = []
        try:
            for name, fn in fwd_hooks:
                r = self.resolve(name)
                if r.point == "input":
                    def pre_hook(_m, args, kwargs, _fn=fn, _name=name):
                        if not args:
                            return None
                        new = _fn(args[0], _name)
                        if new is None:
                            return None
                        return (new,) + tuple(args[1:]), kwargs
                    handles.append(r.module.register_forward_pre_hook(pre_hook, with_kwargs=True))
                else:
                    def post_hook(_m, _args, output, _fn=fn, _name=name):
                        new = _fn(_first_tensor(output), _name)
                        if new is None:
                            return None
                        return _replace_first(output, new)
                    handles.append(r.module.register_forward_hook(post_hook))
            yield
        finally:
            for h in handles:
                h.remove()

    def __call__(self, input: Any, **kwargs: Any):
        return self.forward(input, **kwargs)

    def forward(self, input: Any, *, max_length: int | None = None):
        """Plain forward. Returns the HF output's logits."""
        ids, mask = self._prepare(input, max_length)
        out = self.model(ids, attention_mask=mask)
        return getattr(out, "logits", out)

    def run_with_hooks(
        self, input: Any, fwd_hooks: Iterable[tuple[str, HookFn]],
        *, max_length: int | None = None,
    ):
        with self.hooks(fwd_hooks):
            return self.forward(input, max_length=max_length)

    def run_with_cache(
        self,
        input: Any,
        names: str | Sequence[str] | None = None,
        *,
        stop_after_cache: bool = False,
        cache_device: str | torch.device | None = None,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Run the model and capture activations at ``names``.

        Returns ``(logits, cache)`` where ``cache[name]`` has the natural
        ``(batch, seq, d)`` shape. ``names=None`` caches every non-expert hook
        point (expensive on a 52-layer model — prefer naming what you need).
        ``stop_after_cache=True`` aborts the forward as soon as the deepest
        requested activation is captured (logits comes back None) — this is how
        the nemotron-sae cache pipeline kept 30B forwards cheap.
        """
        if names is None:
            names = self.hook_points()
        if isinstance(names, str):
            names = [names]
        remaining = {n for n in names}
        cache: dict[str, torch.Tensor] = {}

        def make_fn(name: str) -> HookFn:
            def fn(acts: torch.Tensor, _n: str) -> None:
                t = acts.detach()
                if cache_device is not None:
                    t = t.to(cache_device)
                cache[name] = t
                remaining.discard(name)
                if stop_after_cache and not remaining:
                    raise EarlyExit
                return None
            return fn

        logits = None
        try:
            logits = self.run_with_hooks(input, [(n, make_fn(n)) for n in names],
                                         max_length=max_length)
        except EarlyExit:
            pass
        missing = [n for n in names if n not in cache]
        if missing:
            raise RuntimeError(
                f"Hooks at {missing} never fired. If you used stop_after_cache, note "
                f"activations are captured in forward order — a hook *after* the "
                f"deepest requested layer can be cut off."
            )
        return logits, cache

    # -------------------------------------------------------------------- SAEs
    def run_with_saes(
        self,
        input: Any,
        saes: Sequence[Any] | Any,
        *,
        names: Sequence[str] | None = None,
        mode: str = "sae",
        max_length: int | None = None,
    ):
        """Run with SAE reconstructions spliced into the forward pass.

        Each SAE is spliced at ``names[i]`` if given, else at its recorded
        ``metadata['hook_name']`` (the HF module path the published SAEs carry).
        ``mode="zero"`` splices zeros instead — the ablation baseline used to
        normalize delta-CE into "loss recovered".
        """
        if mode not in ("sae", "zero"):
            raise ValueError(f"mode must be 'sae' or 'zero', got {mode!r}")
        saes = [saes] if not isinstance(saes, (list, tuple)) else list(saes)
        if names is None:
            names = []
            for s in saes:
                md = getattr(s.cfg, "metadata", None)
                # dict on nemotron_lens.SAE, attribute-object on sae_lens SAEs.
                hn = md.get("hook_name") if isinstance(md, dict) else getattr(md, "hook_name", None)
                if not hn:
                    raise ValueError(
                        f"{s!r} has no metadata hook_name; pass names=[...] explicitly."
                    )
                names.append(hn)

        def make_fn(sae) -> HookFn:
            def fn(acts: torch.Tensor, _n: str) -> torch.Tensor:
                if mode == "zero":
                    return torch.zeros_like(acts)
                flat = acts.reshape(-1, acts.shape[-1])
                with torch.no_grad():
                    rec = sae.decode(sae.encode(flat.float()))
                return rec.reshape(acts.shape).to(acts.dtype)
            return fn

        return self.run_with_hooks(input, list(zip(names, [make_fn(s) for s in saes])),
                                   max_length=max_length)

    # --------------------------------------------------------------- streaming
    def stream_activations(
        self,
        texts: Iterable[str],
        name: str,
        *,
        batch_size: int = 8,
        context_size: int = 1024,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Iterator[torch.Tensor]:
        """Yield flat ``(n_real_tokens, d)`` activation tensors at hook ``name``
        for batches of ``texts`` — the feed for :func:`nemotron_lens.training.train_sae`.

        Padding positions are dropped using the attention mask, and the forward
        early-exits right after the hooked module, so upper layers + the LM head
        are never computed.
        """
        batch: list[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) < batch_size:
                continue
            yield self._acts_for_batch(batch, name, context_size, device, dtype)
            batch = []
        if batch:
            yield self._acts_for_batch(batch, name, context_size, device, dtype)

    def _acts_for_batch(self, batch, name, context_size, device, dtype) -> torch.Tensor:
        ids, mask = self.to_tokens(batch, max_length=context_size)
        with torch.no_grad():
            _, cache = self.run_with_cache((ids, mask), [name], stop_after_cache=True)
        acts = cache[name]
        if mask is not None and acts.shape[:2] == mask.shape:
            flat = acts[mask.bool()]
        else:  # module outputs that are already flat (e.g. per-expert slices)
            flat = acts.reshape(-1, acts.shape[-1])
        return flat.to(device=device, dtype=dtype)
