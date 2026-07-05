"""nemotron-lens — TransformerLens-style interpretability tooling for NVIDIA
Nemotron hybrid (Mamba-2 + Attention + MoE) models.

Quick start::

    from nemotron_lens import HookedNemotron, SAE

    model = HookedNemotron.from_pretrained()          # Nemotron-3-Nano-30B-A3B
    model.hook_points(kinds=["resid_post"])           # list hookable locations
    logits, cache = model.run_with_cache("Hello", names=["blocks.25.hook_mamba_out"])

    sae = SAE.from_pretrained("Yusser/nemotron-3-nano-30b-a3b-saes",
                              "L2_resid_post/w16384_l0_10")
    feats = sae.encode(cache["blocks.25.hook_mamba_out"].float().flatten(0, 1))

The same SAE repos load with stock ``sae_lens.SAE.from_pretrained`` — the on-disk
format is identical.
"""

from nemotron_lens.hooked_model import DEFAULT_MODEL, EarlyExit, HookedNemotron
from nemotron_lens.sae import SAE, SAEConfig, list_saes
from nemotron_lens.topology import HookSite, dump_topology, enumerate_hooks, load_topology
from nemotron_lens.training import ShuffleBuffer, TrainConfig, TrainingSAE, train_sae

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL",
    "EarlyExit",
    "HookedNemotron",
    "HookSite",
    "SAE",
    "SAEConfig",
    "ShuffleBuffer",
    "TrainConfig",
    "TrainingSAE",
    "dump_topology",
    "enumerate_hooks",
    "list_saes",
    "load_topology",
    "train_sae",
    "__version__",
]
