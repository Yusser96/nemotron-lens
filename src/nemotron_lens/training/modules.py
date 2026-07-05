"""Trainable JumpReLU SAE — the Gemma Scope 2 recipe, as proven on Nemotron.

This is a direct port of the (v3-validated) nemotron-sae training module:
straight-through estimators for the Heaviside gate / L0 / per-latent firing
frequency, learnable per-latent threshold (raw or log parameterization), unit-norm
decoder columns with gradient projection, and a scalar activation-normalization
buffer (``norm_factor``) that travels with the checkpoint.

Layout note: during training ``W_dec`` is stored ``(d_in, d_sae)`` with *columns*
as dictionary directions (GS2 convention). :meth:`TrainingSAE.to_inference` folds
``norm_factor`` + the pre-encoder bias into plain weights and transposes into the
sae_lens inference layout — the same parity-tested fold used to publish the
Nemotron SAEs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nemotron_lens.sae import SAE, SAEConfig

__all__ = ["TrainingSAE", "jump_relu", "l0_norm_diff", "freq_norm_diff"]


class _JumpReLUSTE(torch.autograd.Function):
    """Heaviside H(z − θ) with rectangular-kernel STE for both z and θ (legacy;
    kept for reproducing old runs — the in-window z-kick drove encoder runaway)."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z, theta) = ctx.saved_tensors
        eps = ctx.bandwidth
        in_window = (z - theta).abs() < (eps / 2.0)
        kernel = in_window.to(z.dtype) / eps
        grad_z = grad_out * kernel
        grad_theta = -(grad_out * kernel).sum(dim=0)
        return grad_z, grad_theta, None


class _JumpReLUPaperV3(torch.autograd.Function):
    """f = z·H(z−θ) with the arXiv:2407.14435 v3 pseudo-derivatives (the SAELens
    wiring): ∂f/∂z := H(z−θ) exactly, ∂f/∂θ := −(θ/ε)·K((z−θ)/ε). No STE leak
    into the encoder."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        gate = (z > theta).to(z.dtype)
        return z * gate

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z, theta) = ctx.saved_tensors
        eps = ctx.bandwidth
        grad_z = grad_out * (z > theta).to(z.dtype)
        in_window = ((z - theta).abs() < (eps / 2.0)).to(z.dtype)
        grad_theta = -(theta / eps) * (in_window * grad_out).sum(dim=0)
        return grad_z, grad_theta, None


def jump_relu(z: torch.Tensor, theta: torch.Tensor, bandwidth: float,
              ste_variant: str = "paper_v3") -> torch.Tensor:
    if ste_variant == "paper_v3":
        return _JumpReLUPaperV3.apply(z, theta, bandwidth)
    gate = _JumpReLUSTE.apply(z, theta, bandwidth)
    return z * gate


class _L0STE(torch.autograd.Function):
    """Mean per-token L0 with rectangular-kernel STE on the threshold."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype).sum(dim=-1).mean()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z, theta) = ctx.saved_tensors
        eps = ctx.bandwidth
        in_window = (z - theta).abs() < (eps / 2.0)
        grad_theta = -in_window.to(z.dtype).sum(dim=0) / (eps * z.shape[0])
        return None, grad_out * grad_theta, None


def l0_norm_diff(z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return _L0STE.apply(z, theta, bandwidth)


class _FreqSTE(torch.autograd.Function):
    """Per-latent firing frequency freq_j = mean_b H(z_bj − θ_j), with the STE
    gradient routed into θ_j — feeds GS2's secondary dense-latent penalty."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype).mean(dim=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z, theta) = ctx.saved_tensors
        eps = ctx.bandwidth
        in_window = (z - theta).abs() < (eps / 2.0)
        dfreq_dtheta = -in_window.to(z.dtype).sum(dim=0) / (eps * z.shape[0])
        return None, grad_out * dfreq_dtheta, None


def freq_norm_diff(z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return _FreqSTE.apply(z, theta, bandwidth)


class TrainingSAE(nn.Module):
    """JumpReLU SAE in the GS2 training parameterization (see module docstring)."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        bandwidth: float = 0.001,
        threshold_init: float = 0.001,
        pre_encoder_bias: bool = True,
        freq_bandwidth: float | None = None,
        threshold_param: str = "log",
        decoder_init_norm: float | None = None,
        ste_variant: str = "paper_v3",
        relu_preacts: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.bandwidth = bandwidth
        self.freq_bandwidth = float(freq_bandwidth) if freq_bandwidth is not None else bandwidth
        self.pre_encoder_bias = pre_encoder_bias
        if ste_variant not in ("legacy", "paper_v3"):
            raise ValueError(f"ste_variant must be 'legacy' or 'paper_v3', got {ste_variant!r}")
        self.ste_variant = ste_variant
        # JR App. J: ReLU pre-activations before the gate and every Heaviside STE,
        # so negative pre-activations cannot bias θ-gradients when θ < ε/2.
        self.relu_preacts = relu_preacts
        self.threshold_param = threshold_param

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))  # columns = directions
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if threshold_param == "log":
            if threshold_init <= 0:
                raise ValueError("threshold_param='log' requires threshold_init > 0")
            self.log_theta = nn.Parameter(torch.full((d_sae,), math.log(threshold_init)))
        elif threshold_param == "raw":
            self.register_parameter("theta", nn.Parameter(torch.full((d_sae,), threshold_init)))
        else:
            raise ValueError(f"threshold_param must be 'raw' or 'log', got {threshold_param!r}")

        # x_norm = x * norm_factor; the trainer sets this before step 1 so inputs
        # live in the space GS2's ε/θ₀/λ constants are calibrated to, and it is
        # persisted in checkpoints so inference needs no external normalization.
        self.register_buffer("norm_factor", torch.tensor(1.0))

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))
            if decoder_init_norm is not None:
                self.W_dec.mul_(float(decoder_init_norm))
            self.W_enc.copy_(self.W_dec)  # tied at init (after scaling), untied after

    # -------------------------------------------------------------- parameters
    @property
    def theta(self) -> torch.Tensor:
        """Effective threshold: exp(log_theta) in log mode (autograd chains
        through the exp), the raw Parameter otherwise."""
        if getattr(self, "threshold_param", "raw") == "log":
            return self.log_theta.exp()
        p = self._parameters.get("theta")
        if p is None:
            raise AttributeError("theta")
        return p

    # ----------------------------------------------------------------- forward
    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.norm_factor
        if self.pre_encoder_bias:
            x = x - self.b_dec
        return x @ self.W_enc + self.b_enc

    def gate_preacts(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_pre(x)
        return torch.relu(z) if self.relu_preacts else z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gate_preacts(x)
        return jump_relu(z, self.theta, self.bandwidth, self.ste_variant)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return (f @ self.W_dec.T + self.b_dec) / self.norm_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    def l0(self, x: torch.Tensor) -> torch.Tensor:
        return l0_norm_diff(self.gate_preacts(x), self.theta, self.bandwidth)

    def frequency(self, x: torch.Tensor) -> torch.Tensor:
        return freq_norm_diff(self.gate_preacts(x), self.theta, self.freq_bandwidth)

    @torch.no_grad()
    def hard_l0(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gate_preacts(x)
        return (z > self.theta).to(z.dtype).sum(dim=-1).mean()

    # ------------------------------------------------- decoder-norm constraints
    @torch.no_grad()
    def renormalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        if self.W_dec.grad is None:
            return
        dot = (self.W_dec.grad * self.W_dec).sum(dim=0, keepdim=True)
        self.W_dec.grad.sub_(dot * self.W_dec)

    # -------------------------------------------------------------------- fold
    @torch.no_grad()
    def to_inference(self, unit_norm_decoder: bool = True,
                     metadata: dict | None = None) -> SAE:
        """Fold ``norm_factor`` + pre-encoder bias into plain weights and return a
        standard :class:`nemotron_lens.SAE` on **raw** activations (sae_lens layout).

        For a raw activation x this SAE computed
        ``f = JumpReLU((x·c − b_dec) @ W_enc + b_enc)``, ``x̂ = (f @ W_dec.T + b_dec)/c``.
        The fold (valid for θ ≥ 0, guaranteed in log mode)::

            We = c·W_enc, be = b_enc − b_dec@W_enc, Wd = W_dec.T/c, bd = b_dec/c, thr = θ

        ``unit_norm_decoder=True`` additionally rescales decoder rows to unit norm
        (function-preserving; SAEBench's probe evals expect it).
        """
        theta = self.theta.detach()
        if not bool((theta >= 0).all()):
            raise RuntimeError("theta has negative entries; the fold requires theta >= 0.")
        c = float(self.norm_factor)
        We = (c * self.W_enc).detach().clone()
        be = (self.b_enc - self.b_dec @ self.W_enc).detach().clone()
        Wd = (self.W_dec.T / c).detach().clone()
        bd = (self.b_dec / c).detach().clone()
        thr = theta.clone()
        if unit_norm_decoder:
            s = Wd.norm(dim=1).clamp_min(1e-8)
            Wd = Wd / s[:, None]
            We = We * s[None, :]
            be = be * s
            thr = thr * s

        cfg = SAEConfig(d_in=self.d_in, d_sae=self.d_sae, metadata=dict(metadata or {}))
        sae = SAE(cfg)
        sae.load_state_dict({
            "W_enc": We.contiguous(), "b_enc": be.contiguous(),
            "W_dec": Wd.contiguous(), "b_dec": bd.contiguous(),
            "threshold": thr.contiguous(),
        })
        return sae
