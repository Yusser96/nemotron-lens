"""Bridge a Nemotron HF model into SAE-Lens / SAE-Bench (TransformerLens-shaped).

TransformerLens does not support hybrid Mamba-2 + attention + MoE models, but
SAE-Lens ships ``HookedProxyLM`` — a ``HookedRootModule`` wrapper over any HF
causal LM whose hook names are plain module paths. This module (a port of the
battle-tested shim from nemotron-sae's external_eval) turns a
:class:`~nemotron_lens.HookedNemotron` into a proxy that satisfies BOTH
libraries, including SAE-Bench's ``isinstance(model, HookedTransformer)``
runtime checks and its habit of passing raw strings to ``run_with_cache``.

Requires the ``eval`` extra: ``pip install nemotron-lens[eval]``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import torch

log = logging.getLogger(__name__)

__all__ = ["build_proxy", "inject_proxy_as_tlens"]

# Captured once: the stock HookedProxyLM.forward, delegated to for tensor inputs.
_BASE_HOOKEDPROXY_FORWARD = None


def build_proxy(model, tokenizer, hook_names, model_name: str | None = None):
    """Wrap an HF causal LM as a ``HookedProxyLM`` restricted to ``hook_names``
    (module paths). Restricting the hooked submodules keeps setup cheap on 30B."""
    from sae_lens.load_model import HookedProxyLM

    proxy = HookedProxyLM(model, tokenizer, hook_names=list(hook_names))
    _make_saebench_compatible(proxy, model, model_name)
    return proxy


def _encode_batch(self, inputs, max_length=1024):
    """Tokenize str / list[str] to (input_ids, attention_mask) on the model device.
    LEFT padding: Mamba layers process the sequence causally, so real tokens must
    be right-aligned (SAE-Bench's absorption reads end-relative positions)."""
    tok = self.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(inputs, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
    finally:
        tok.padding_side = prev_side
    device = next(self.model.parameters()).device
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _tokenizing_forward(self, tokens, return_type="logits", loss_per_token=False,
                        stop_at_layer=None, _names_filter=None, **kwargs):
    """``forward`` override: tokenize raw str/list[str] input; tensors take the
    stock ``HookedProxyLM.forward`` path (incl. its early-stop logic)."""
    base_forward = _BASE_HOOKEDPROXY_FORWARD
    if isinstance(tokens, torch.Tensor):
        return base_forward(self, tokens, return_type=return_type,
                            loss_per_token=loss_per_token, stop_at_layer=stop_at_layer,
                            _names_filter=_names_filter, **kwargs)

    from sae_lens.load_model import (
        Output,
        StopForward,
        StopManager,
        _extract_logits_from_output,
        _normalize_hook_names,
        lm_cross_entropy_loss,
    )

    input_ids, attn_mask = _encode_batch(self, tokens)

    stop_names = _normalize_hook_names(_names_filter, self.named_modules_dict)
    stop_hooks = []
    if stop_names and return_type == "logits":
        stop_manager = StopManager(stop_names)
        for hook_name in stop_names:
            module = self.named_modules_dict[hook_name]
            stop_hooks.append(module.register_forward_hook(stop_manager.get_stop_hook_fn(hook_name)))
    try:
        output = self.model(input_ids, attention_mask=attn_mask)
        logits = _extract_logits_from_output(output)
    except StopForward:
        return None
    finally:
        for stop_hook in stop_hooks:
            stop_hook.remove()

    if return_type == "logits":
        return logits
    if logits is None:
        raise ValueError("return_type='both' requires logits, but logits=None.")
    loss = lm_cross_entropy_loss(logits, input_ids, per_token=loss_per_token)
    return Output(logits, loss)


def _make_saebench_compatible(proxy, model, model_name) -> None:
    """Make the proxy pass SAE-Bench's three HookedTransformer assumptions:
    (1) ``isinstance(model, HookedTransformer)`` beartype gates — reassign
    ``__class__`` to a ``(HookedProxyLM, HookedTransformer)`` subclass (both share
    the HookedRootModule ancestor; HookedTransformer.__init__ never runs);
    (2) ``model.cfg.{model_name,device,d_model}`` reads — attach a namespace;
    (3) raw-string ``run_with_cache`` calls — the forward override above."""
    global _BASE_HOOKEDPROXY_FORWARD
    from transformer_lens import HookedTransformer

    if not isinstance(proxy, HookedTransformer):
        base = type(proxy)
        if _BASE_HOOKEDPROXY_FORWARD is None:
            _BASE_HOOKEDPROXY_FORWARD = base.forward
        hooked_cls = type(f"{base.__name__}AsHooked", (base, HookedTransformer),
                          {"forward": _tokenizing_forward})
        proxy.__class__ = hooked_cls

    proxy.cfg = SimpleNamespace(
        model_name=model_name or getattr(model.config, "_name_or_path", "nemotron-proxy"),
        device=str(next(model.parameters()).device),
        d_model=getattr(model.config, "hidden_size", None),
    )


@contextmanager
def inject_proxy_as_tlens(proxy):
    """Monkeypatch ``HookedTransformer.from_pretrained_no_processing`` to return
    ``proxy`` — for SAE-Bench evals that load the model by name internally
    (absorption). Restores the original on exit."""
    import transformer_lens

    ht = transformer_lens.HookedTransformer
    original = ht.from_pretrained_no_processing
    ht.from_pretrained_no_processing = staticmethod(lambda *a, **k: proxy)  # type: ignore[assignment]
    try:
        yield
    finally:
        ht.from_pretrained_no_processing = original  # type: ignore[assignment]
