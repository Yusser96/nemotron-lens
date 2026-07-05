"""Run SAE-Bench and SAE-Lens evaluations on a Nemotron SAE.

Port of nemotron-sae's ``external_eval`` harness. SAE-Bench's high-level runner
hard-gates models to pythia/gemma, so each eval's own entry point is driven
directly with our proxy model:

- ``core``           — L0, FVU/explained-variance, CE-loss-recovered, KL (the
                       Gemma-Scope-comparable numbers)
- ``sparse_probing`` — probe accuracy from top-k SAE latents
- ``scr`` / ``tpp``  — spurious-correlation removal / targeted-probe perturbation
- ``absorption``     — feature absorption (model injected via monkeypatch)

plus ``run_saelens_evals`` for ``sae_lens.run_evals`` (same metric family,
independent implementation).

Requires ``pip install nemotron-lens[eval]`` and, for SAE-Bench, the
``sae-bench`` package. Datasets: ``"org/name"`` or ``"org/name:config"``.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Sequence

import torch

from nemotron_lens.evals.proxy import build_proxy, inject_proxy_as_tlens
from nemotron_lens.hooked_model import HookedNemotron
from nemotron_lens.sae import SAE

log = logging.getLogger(__name__)

__all__ = ["run_saebench", "run_saelens_evals", "ALL_EVALS", "DEFAULT_DATASET"]

ALL_EVALS = ["core", "sparse_probing", "scr", "tpp", "absorption"]
# Loads cleanly under datasets>=4 with no config arg; standard for SAE evals.
DEFAULT_DATASET = "NeelNanda/pile-10k"


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.tolist()
    if hasattr(x, "__dict__") and not isinstance(x, (str, int, float, bool)):
        try:
            return {k: _to_jsonable(v) for k, v in vars(x).items()}
        except Exception:
            return str(x)
    return x


def _mk_config(cls, **kw):
    """Instantiate an SAE-Bench eval config, setting only fields it actually has
    (defensive against minor SAE-Bench version drift)."""
    cfg = cls()
    for k, v in kw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def resolve_dataset(spec: str, *, streaming: bool = True):
    """``org/name`` passes through; ``org/name:config`` pre-loads a streaming
    dataset object (sae_lens's string path cannot carry a dataset config)."""
    if ":" not in spec or "://" in spec:
        return spec
    path, _, config = spec.partition(":")
    from datasets import load_dataset

    return load_dataset(path, config, split="train", streaming=streaming)


def _hook_name_of(sae: SAE, hook_name: str | None) -> str:
    hn = hook_name or sae.metadata.get("hook_name")
    if not hn:
        raise ValueError("SAE metadata has no hook_name; pass hook_name=... explicitly.")
    return hn


def _hook_layer_of(sae: SAE, hn: str) -> int:
    hl = sae.metadata.get("hook_layer") or sae.metadata.get("layer")
    if hl is not None:
        return int(hl)
    import re

    m = re.search(r"\.(\d+)(?:\.|$)", hn)
    return int(m.group(1)) if m else 0


def _to_saebench_sae(sae: SAE, *, model_name, hook_name, hook_layer, device, dtype,
                     context_size):
    """nemotron_lens.SAE -> sae_bench.custom_saes.jumprelu_sae.JumpReluSAE
    (identical math + state-dict keys; SAE dtype must match the LLM compute dtype)."""
    from sae_bench.custom_saes.jumprelu_sae import JumpReluSAE

    bench = JumpReluSAE(d_in=sae.d_in, d_sae=sae.d_sae, model_name=model_name,
                        hook_layer=hook_layer, device=device, dtype=dtype,
                        hook_name=hook_name)
    result = bench.load_state_dict(sae.state_dict(), strict=False)
    missing = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"} & set(result.missing_keys)
    if missing:
        raise RuntimeError(f"SAE-Bench SAE failed to load keys {missing}")
    bench = bench.to(device=device, dtype=dtype)
    bench.cfg.architecture = "jumprelu"
    bench.cfg.normalize_activations = "none"
    bench.cfg.hook_name = hook_name
    bench.cfg.hook_layer = hook_layer
    bench.cfg.context_size = context_size
    bench.cfg.dtype = str(dtype).split(".")[-1]
    return bench


def _build_store(proxy, sl_sae, *, dataset, context_size, batch_size, device):
    from sae_lens import ActivationsStore

    return ActivationsStore.from_sae(
        model=proxy, sae=sl_sae, dataset=resolve_dataset(dataset),
        dataset_trust_remote_code=True, context_size=context_size, streaming=True,
        store_batch_size_prompts=batch_size, n_batches_in_buffer=8,
        train_batch_size_tokens=4096, device=device,
    )


# ------------------------------------------------------------------ sae_lens
def run_saelens_evals(
    sae: SAE,
    model: HookedNemotron,
    *,
    hook_name: str | None = None,
    dataset: str = DEFAULT_DATASET,
    context_size: int = 1024,
    llm_batch_size: int = 2,   # KL runs a full-vocab softmax; small batches only
    n_recon_batches: int = 10,
    n_sparsity_batches: int = 4,
    out_dir: str | Path | None = None,
) -> dict:
    """``sae_lens.run_evals`` against the proxy: L0, explained variance / MSE /
    cossim, CE-loss-recovered, KL, shrinkage."""
    from sae_lens.evals import get_eval_everything_config, run_evals
    from sae_lens.training.activation_scaler import ActivationScaler

    hn = _hook_name_of(sae, hook_name)
    device = str(model.device)
    model_name = sae.metadata.get("model_name", "nemotron")
    proxy = build_proxy(model.model, model.tokenizer, [hn], model_name=model_name)

    sl_sae = sae.to_saelens(device=device)
    sl_sae.cfg.metadata.hook_name = hn
    store = _build_store(proxy, sl_sae, dataset=dataset, context_size=context_size,
                         batch_size=llm_batch_size, device=device)
    eval_cfg = get_eval_everything_config(
        batch_size_prompts=llm_batch_size,
        n_eval_reconstruction_batches=n_recon_batches,
        n_eval_sparsity_variance_batches=n_sparsity_batches,
    )
    scalar, _feature = run_evals(sae=sl_sae, activation_store=store, model=proxy,
                                 activation_scaler=ActivationScaler(),
                                 eval_config=eval_cfg, verbose=True)
    result = {"library": "sae_lens", "hook_name": hn, "dataset": dataset,
              "metrics": _to_jsonable(scalar)}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sae_lens_evals.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# ----------------------------------------------------------------- sae_bench
def run_saebench(
    sae: SAE,
    model: HookedNemotron,
    evals: Sequence[str] = ("core",),
    *,
    hook_name: str | None = None,
    dataset: str = DEFAULT_DATASET,
    context_size: int = 1024,
    llm_batch_size: int = 8,
    n_recon_batches: int = 10,
    n_sparsity_batches: int = 4,
    out_dir: str | Path = "saebench_out",
    sae_name: str = "nemotron_sae",
) -> dict:
    """Run the selected SAE-Bench evals; one eval failing never sinks the rest.
    Results (+ per-eval JSON artifacts) land under ``out_dir``."""
    eval_types = [e for e in evals if e in ALL_EVALS]
    out = Path(out_dir)
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    hn = _hook_name_of(sae, hook_name)
    hook_layer = _hook_layer_of(sae, hn)
    device = str(model.device)
    llm_dtype = next(model.model.parameters()).dtype
    model_name = sae.metadata.get("model_name", "nemotron")
    proxy = build_proxy(model.model, model.tokenizer, [hn], model_name=model_name)

    # SAE-Bench encodes the model's (bf16) activations with no cast — match dtypes.
    bench_sae = _to_saebench_sae(sae, model_name=model_name, hook_name=hn,
                                 hook_layer=hook_layer, device=device,
                                 dtype=llm_dtype, context_size=context_size)

    def _core() -> dict:
        from sae_bench.evals.core.eval_config import CoreEvalConfig
        from sae_bench.evals.core.main import run_evals as core_run_evals

        sl_sae = sae.to_saelens(device=device)
        sl_sae.cfg.metadata.hook_name = hn
        store = _build_store(proxy, sl_sae, dataset=dataset, context_size=context_size,
                             batch_size=llm_batch_size, device=device)
        cfg = _mk_config(
            CoreEvalConfig, model_name=model_name, llm_dtype=str(llm_dtype).split(".")[-1],
            batch_size_prompts=llm_batch_size, dataset=dataset, context_size=context_size,
            n_eval_reconstruction_batches=n_recon_batches,
            n_eval_sparsity_variance_batches=n_sparsity_batches,
            compute_kl=True, compute_ce_loss=True, compute_l2_norms=True,
            compute_sparsity_metrics=True, compute_variance_metrics=True,
        )
        scalar, _ = core_run_evals(bench_sae, store, proxy, eval_config=cfg, verbose=True)
        return scalar

    def _sparse_probing() -> dict:
        from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
        from sae_bench.evals.sparse_probing.main import run_eval_single_sae

        cfg = _mk_config(SparseProbingEvalConfig, model_name=model_name,
                         llm_dtype=str(llm_dtype).split(".")[-1],
                         llm_batch_size=llm_batch_size, context_length=context_size,
                         lower_vram_usage=True)
        res, _ = run_eval_single_sae(cfg, bench_sae, proxy, device, str(artifacts))
        return res

    def _scr_tpp(perform_scr: bool) -> dict:
        from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
        from sae_bench.evals.scr_and_tpp.main import run_eval_single_sae

        cfg = _mk_config(ScrAndTppEvalConfig, model_name=model_name,
                         llm_dtype=str(llm_dtype).split(".")[-1],
                         llm_batch_size=llm_batch_size, context_length=context_size,
                         perform_scr=perform_scr, lower_vram_usage=True)
        res, _ = run_eval_single_sae(cfg, bench_sae, proxy, device, str(artifacts))
        return res

    def _absorption() -> dict:
        from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
        from sae_bench.evals.absorption.main import run_eval as absorption_run_eval

        cfg = _mk_config(AbsorptionEvalConfig, model_name=model_name,
                         llm_dtype=str(llm_dtype).split(".")[-1],
                         llm_batch_size=llm_batch_size)
        ap = out / "absorption_out"
        ap.mkdir(parents=True, exist_ok=True)
        with inject_proxy_as_tlens(proxy):
            absorption_run_eval(cfg, [(sae_name, bench_sae)], device, str(ap))
        merged = {}
        for f in sorted(ap.rglob("*.json")):
            try:
                merged[str(f)] = json.loads(f.read_text())
            except Exception:
                merged[str(f)] = "<unreadable>"
        return merged or {"note": "absorption produced no JSON output"}

    runners = {
        "core": _core,
        "sparse_probing": _sparse_probing,
        "scr": lambda: _scr_tpp(True),
        "tpp": lambda: _scr_tpp(False),
        "absorption": _absorption,
    }
    results: dict = {}
    for ev in eval_types:
        try:
            log.info("=== sae_bench:%s ===", ev)
            results[ev] = runners[ev]()
        except Exception as e:
            log.exception("sae_bench:%s failed", ev)
            results[ev] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            # lower_vram_usage can leave the 30B model parked on CPU after a
            # mid-eval exception (Mamba mixers hard-require CUDA) — restore it,
            # and reclaim fragmented VRAM between evals.
            try:
                proxy.to(device)
            except Exception:
                log.exception("could not restore model to %s", device)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {"library": "sae_bench", "hook_name": hn, "eval_types": list(eval_types),
               "metrics": _to_jsonable(results)}
    (out / f"{sae_name}.json").write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote %s", out / f"{sae_name}.json")
    return payload
