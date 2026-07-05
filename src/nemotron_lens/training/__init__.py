"""JumpReLU SAE training (Gemma Scope 2 recipe, Nemotron-validated defaults)."""

from nemotron_lens.training.modules import TrainingSAE
from nemotron_lens.training.trainer import (
    ShuffleBuffer,
    TrainConfig,
    estimate_norm_factor,
    train_sae,
)

__all__ = ["TrainingSAE", "TrainConfig", "ShuffleBuffer", "estimate_norm_factor", "train_sae"]
