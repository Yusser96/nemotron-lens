"""SAE evaluation harnesses (SAE-Bench + SAE-Lens). Requires the ``eval`` extra:
``pip install nemotron-lens[eval]`` (sae-lens, transformer-lens, sae-bench)."""

from nemotron_lens.evals.proxy import build_proxy, inject_proxy_as_tlens
from nemotron_lens.evals.saebench import (
    ALL_EVALS,
    DEFAULT_DATASET,
    run_saebench,
    run_saelens_evals,
)

__all__ = [
    "ALL_EVALS",
    "DEFAULT_DATASET",
    "build_proxy",
    "inject_proxy_as_tlens",
    "run_saebench",
    "run_saelens_evals",
]
