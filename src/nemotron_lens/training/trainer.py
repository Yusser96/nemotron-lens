"""SAE training loop — the Gemma Scope 2 recipe as validated on Nemotron-3-Nano.

Direct port of the nemotron-sae v3 trainer (the configuration that first reached
its L0 target on this model), minus the disk shard-cache: activations come from
any object with ``next_batch() -> (batch_size, d_in)`` — use :class:`ShuffleBuffer`
to adapt an iterator of activation chunks (e.g.
``HookedNemotron.stream_activations``).

Loss::

    L = ||x − x̂||²  +  λ(t) · c(L0*) · (||f||₀ − L0*)²  +  λ_f(t) · Σ_j relu(freq_j − f*)²

with c = 1/(2·L0*) (GS2 Eq. 5) or 2/L0* (legacy), λ/λ_f linearly warmed up, LR
linearly warmed 0.1η→η then held constant, Adam(0, 0.999), unit-norm decoder
columns with tangent-space gradient projection, and activations normalized by a
scalar ``norm_factor`` (E[||x||²]=1, GS1 App. A) that is estimated once at start
and persisted in every checkpoint.

Training telemetry goes to ``train_log.jsonl`` (one line per ``log_every`` steps)
including the threshold-health fields that catch the known JumpReLU failure modes
live (STE window occupancy, z_std, θ stats, windowed dead %).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol

import torch
from safetensors.torch import load_file, save_file

from nemotron_lens.training.modules import TrainingSAE

log = logging.getLogger(__name__)

__all__ = ["TrainConfig", "ShuffleBuffer", "train_sae", "estimate_norm_factor"]

TRAINER_STATE_FILENAME = "trainer_state.pt"


class BufferLike(Protocol):
    def next_batch(self) -> torch.Tensor: ...


@dataclass
class TrainConfig:
    """Defaults = the nemotron-sae prod_v3 recipe (the first healthy config).

    Change ``l0_target`` / ``d_sae`` / ``n_steps`` freely; the gate/threshold
    constants (bandwidth, threshold_init) are calibrated to the "vector" norm
    convention (E[||x||²]=1) — change them together or not at all.
    """

    # sweep point
    d_sae: int = 16384
    l0_target: int = 10
    # optimization (GS2 §3.2)
    lr: float = 7.0e-5
    batch_size: int = 4096
    n_steps: int = 200_000
    warmup_steps: int = 1_000
    l0_warmup_steps: int = 50_000
    adam_beta1: float = 0.0
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    grad_clip: float | None = 1.0
    # JumpReLU gate (JR App. I/J; valid under norm_convention="vector")
    bandwidth: float = 0.001
    threshold_param: str = "log"
    threshold_init: float = 0.001
    threshold_lr_mult: float = 1.0
    freq_bandwidth: float | None = None
    ste_variant: str = "paper_v3"
    relu_preacts: bool = True
    quad_form: str = "gs2"
    # decoder constraint (GS2 §3.2)
    decoder_unit_norm: bool = True
    decoder_init_norm: float | None = None
    # activation normalization
    norm_convention: str = "vector"      # E[||x||²]=1 (GS1 App. A) | "coord_rms"
    # secondary dense-latent penalty (GS2 §2.3) — off by default
    dead_freq_threshold: float = 0.05
    freq_coeff: float = 0.0
    freq_warmup_steps: int = 50_000
    # io / telemetry
    log_every: int = 100
    ckpt_every: int = 5_000
    dead_win_steps: int = 1_000


@dataclass
class StepLog:
    step: int
    loss: float
    mse: float
    l0_penalty: float
    hard_l0: float
    lr: float
    lambda_l0: float
    dead_pct: float
    freq_penalty: float
    lambda_freq: float
    max_freq: float
    pct_dense: float
    theta_median: float = float("nan")
    theta_min: float = float("nan")
    z_std: float = float("nan")
    window_occ_pct: float = float("nan")
    dead_win_pct: float = float("nan")
    enc_norm_mean: float = float("nan")
    dec_norm_mean: float = float("nan")


# ------------------------------------------------------------------- schedules
def lr_warmup_constant(step: int, peak_lr: float, warmup_steps: int) -> float:
    """Linear warmup 0.1·η → η, then constant (GS2 §3.2). Independent of n_steps
    so resuming with a larger budget never jumps the LR."""
    if step < warmup_steps:
        return peak_lr * (0.1 + 0.9 * step / max(1, warmup_steps))
    return peak_lr


def linear_warmup(step: int, peak: float, warmup_steps: int) -> float:
    return peak if step >= warmup_steps else peak * (step / max(1, warmup_steps))


def quad_l0_loss(l0_diff: torch.Tensor, l0_target: int, form: str = "gs2") -> torch.Tensor:
    if form == "gs2":
        coeff = 1.0 / (2.0 * max(l0_target, 1))
    elif form == "legacy":
        coeff = 2.0 / max(l0_target, 1)
    else:
        raise ValueError(f"quad_form must be 'legacy' or 'gs2', got {form!r}")
    return coeff * (l0_diff - float(l0_target)).pow(2)


def freq_penalty(freq: torch.Tensor, freq_target: float) -> torch.Tensor:
    """One-sided hinge²: only latents firing above f* are pushed down."""
    return torch.relu(freq - float(freq_target)).pow(2).sum()


# ---------------------------------------------------------------------- buffer
class ShuffleBuffer:
    """Adapt an iterator of activation chunks ``(n, d_in)`` into shuffled
    fixed-size training batches (the rolling-buffer scheme from nemotron-sae:
    keep ``n_batches_in_buffer × batch_size`` rows, refill + reshuffle when the
    buffer drops below half)."""

    def __init__(self, source: Iterable[torch.Tensor], batch_size: int,
                 n_batches_in_buffer: int = 8, seed: int = 0):
        self._source: Iterator[torch.Tensor] = iter(source)
        self.batch_size = batch_size
        self.capacity = batch_size * n_batches_in_buffer
        self._store: torch.Tensor | None = None  # (n, d) rows, pre-shuffled
        self._pos = 0
        self._exhausted = False
        self._gen = torch.Generator().manual_seed(seed)

    def _remaining(self) -> int:
        return 0 if self._store is None else self._store.shape[0] - self._pos

    def _refill(self) -> None:
        parts = [] if self._store is None else [self._store[self._pos:]]
        n = sum(int(p.shape[0]) for p in parts)
        while n < self.capacity and not self._exhausted:
            try:
                chunk = next(self._source)
            except StopIteration:
                self._exhausted = True
                break
            if chunk.ndim != 2:
                chunk = chunk.reshape(-1, chunk.shape[-1])
            parts.append(chunk.detach())
            n += int(chunk.shape[0])
        if not parts:
            self._store, self._pos = None, 0
            return
        store = torch.cat(parts, dim=0)
        perm = torch.randperm(store.shape[0], generator=self._gen)
        self._store, self._pos = store[perm], 0

    def next_batch(self) -> torch.Tensor:
        if self._remaining() < max(self.batch_size, self.capacity // 2) and not self._exhausted:
            self._refill()
        if self._remaining() < self.batch_size:
            if self._exhausted and self._remaining() == 0:
                raise StopIteration
            if self._exhausted:  # final partial batch
                out = self._store[self._pos:]
                self._pos = self._store.shape[0]
                return out
            self._refill()
            if self._remaining() == 0:
                raise StopIteration
        out = self._store[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return out

    @torch.no_grad()
    def estimate_norm_factor(self, d_in: int, convention: str = "vector",
                             n_tokens: int = 65_536) -> float:
        """Estimate the normalization scalar from buffered rows *without*
        consuming them (fills the buffer if needed)."""
        if self._remaining() < min(n_tokens, self.capacity) and not self._exhausted:
            self._refill()
        if self._store is None or self._remaining() == 0:
            return 1.0
        rows = self._store[self._pos:self._pos + n_tokens].to(torch.float32)
        return estimate_norm_factor(rows, d_in=d_in, convention=convention)


def estimate_norm_factor(x: torch.Tensor, d_in: int, convention: str = "vector") -> float:
    """c such that x·c matches the convention: "vector" → E[||x||²]=1 (GS1 App. A,
    what the GS2 constants assume); "coord_rms" → E[||x||²]=d_in."""
    mean_sq_norm = float(x.to(torch.float32).pow(2).sum(dim=-1).mean())
    if mean_sq_norm <= 0.0:
        return 1.0
    if convention == "vector":
        return math.sqrt(1.0 / mean_sq_norm)
    if convention == "coord_rms":
        return math.sqrt(d_in / mean_sq_norm)
    raise ValueError(f"norm_convention must be 'coord_rms' or 'vector', got {convention!r}")


# --------------------------------------------------------------------- resume
def _find_latest_checkpoint(out_dir: Path) -> tuple[int, Path] | None:
    ckpts = sorted(out_dir.glob("sae_step_*.safetensors"))
    if not ckpts:
        return None
    latest = ckpts[-1]
    try:
        return int(latest.stem.split("_")[-1]), latest
    except ValueError:
        return None


def _save_checkpoint(sae, optim, step, ever_fired, steps_since_fired, out_dir: Path) -> Path:
    ckpt_path = out_dir / f"sae_step_{step:07d}.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in sae.state_dict().items()},
              str(ckpt_path))
    tmp = (out_dir / TRAINER_STATE_FILENAME).with_suffix(".pt.tmp")
    torch.save(
        {
            "step": step,
            "optimizer": optim.state_dict(),
            "ever_fired": ever_fired.detach().cpu(),
            "steps_since_fired": (steps_since_fired.detach().cpu()
                                  if steps_since_fired is not None else None),
        },
        tmp,
    )
    tmp.replace(out_dir / TRAINER_STATE_FILENAME)  # atomic: no half-written state
    return ckpt_path


# ----------------------------------------------------------------------- train
def train_sae(
    buffer: BufferLike | Iterable[torch.Tensor],
    d_in: int,
    out_dir: str | Path,
    cfg: TrainConfig = TrainConfig(),
    *,
    norm_factor: float | Callable[[], float] | None = None,
    device: str | None = None,
) -> Path:
    """Train a JumpReLU SAE; returns the final checkpoint path.

    ``buffer`` is anything with ``next_batch() -> (batch_size, d_in)`` — a
    :class:`ShuffleBuffer`, or any iterator of ``(n, d_in)`` chunks (wrapped
    automatically). Auto-resumes from the latest ``sae_step_*.safetensors`` in
    ``out_dir``. ``norm_factor``: explicit float, a callable, or None to estimate
    from the buffer (ShuffleBuffer) / default 1.0.
    """
    if not hasattr(buffer, "next_batch"):
        buffer = ShuffleBuffer(buffer, batch_size=cfg.batch_size)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Training JumpReLU SAE: d_in=%d d_sae=%d L0*=%d steps=%d device=%s",
             d_in, cfg.d_sae, cfg.l0_target, cfg.n_steps, device)

    sae = TrainingSAE(
        d_in=d_in, d_sae=cfg.d_sae, bandwidth=cfg.bandwidth,
        threshold_init=cfg.threshold_init, freq_bandwidth=cfg.freq_bandwidth,
        threshold_param=cfg.threshold_param, decoder_init_norm=cfg.decoder_init_norm,
        ste_variant=cfg.ste_variant, relu_preacts=cfg.relu_preacts,
    ).to(device)

    if cfg.threshold_lr_mult != 1.0:
        theta_names = {"theta", "log_theta"}
        groups = [
            {"params": [p for n, p in sae.named_parameters() if n not in theta_names],
             "lr_mult": 1.0},
            {"params": [p for n, p in sae.named_parameters() if n in theta_names],
             "lr_mult": cfg.threshold_lr_mult},
        ]
        optim = torch.optim.Adam(groups, lr=cfg.lr,
                                 betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)
    else:
        optim = torch.optim.Adam(sae.parameters(), lr=cfg.lr,
                                 betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)

    ever_fired = torch.zeros(cfg.d_sae, dtype=torch.bool, device=device)
    steps_since_fired = (torch.zeros(cfg.d_sae, dtype=torch.int32, device=device)
                         if cfg.dead_win_steps > 0 else None)

    # ---- resume ------------------------------------------------------------
    start_step = 0
    latest = _find_latest_checkpoint(out_dir)
    if latest is not None:
        last_step, ckpt_path = latest
        log.info("Resuming from %s (step=%d)", ckpt_path, last_step)
        sae.load_state_dict(load_file(str(ckpt_path)))
        state_path = out_dir / TRAINER_STATE_FILENAME
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=False)
            try:
                optim.load_state_dict(state["optimizer"])
            except ValueError as e:
                raise RuntimeError(
                    "Optimizer state does not match the param-group layout — "
                    "threshold_lr_mult probably changed on a resumed run. Restore it "
                    "or start a fresh out_dir."
                ) from e
            ever_fired = state["ever_fired"].to(device).bool()
            ssf = state.get("steps_since_fired")
            if ssf is not None and steps_since_fired is not None:
                steps_since_fired = ssf.to(device).to(torch.int32)
        else:
            log.warning("No %s next to %s; resuming weights only.", TRAINER_STATE_FILENAME, ckpt_path)
        start_step = last_step

    # ---- normalization -------------------------------------------------------
    if start_step == 0:
        if callable(norm_factor):
            c = float(norm_factor())
        elif norm_factor is not None:
            c = float(norm_factor)
        elif hasattr(buffer, "estimate_norm_factor"):
            c = float(buffer.estimate_norm_factor(d_in=d_in, convention=cfg.norm_convention))
        else:
            c = 1.0
            log.warning("No norm_factor and the buffer cannot estimate one; using 1.0. "
                        "GS2's bandwidth/threshold constants assume E[||x||^2]=1.")
        sae.norm_factor.fill_(c)
        log.info("norm_factor=%.6f (convention=%s)", c, cfg.norm_convention)
    else:
        log.info("Reusing norm_factor=%.6f from checkpoint", float(sae.norm_factor))

    if start_step >= cfg.n_steps:
        log.info("Nothing to train (resume step %d >= n_steps %d).", start_step, cfg.n_steps)
        return out_dir / f"sae_step_{cfg.n_steps:07d}.safetensors"

    log_f = open(out_dir / "train_log.jsonl", "a" if start_step > 0 else "w")
    t0 = time.time()
    norm_factor_sq = sae.norm_factor.pow(2)
    final_step = cfg.n_steps
    # Health-alarm state: the two known JumpReLU collapse signatures (log-only).
    occ_low_since: int | None = None
    occ_alarm = False
    zstd_hist: list[tuple[int, float]] = []
    zstd_alarm = False

    for step in range(start_step + 1, cfg.n_steps + 1):
        try:
            x = buffer.next_batch().to(device, dtype=torch.float32)
        except StopIteration:
            final_step = step - 1
            log.warning("Activation source exhausted at step %d/%d; stopping early.",
                        final_step, cfg.n_steps)
            break

        lr = lr_warmup_constant(step, cfg.lr, cfg.warmup_steps)
        lam = linear_warmup(step, peak=1.0, warmup_steps=cfg.l0_warmup_steps)
        lam_f = linear_warmup(step, peak=cfg.freq_coeff, warmup_steps=cfg.freq_warmup_steps)
        for g in optim.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

        x_hat, f = sae(x)
        # MSE in normalized space (decode returns raw scale, so scale back by c²).
        mse = (x - x_hat).pow(2).mean() * norm_factor_sq
        freq = sae.frequency(x)
        l0_diff = freq.sum() if sae.freq_bandwidth == sae.bandwidth else sae.l0(x)
        l0_pen = quad_l0_loss(l0_diff, cfg.l0_target, cfg.quad_form)
        f_pen = freq_penalty(freq, cfg.dead_freq_threshold)
        loss = mse + lam * l0_pen + lam_f * f_pen

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.decoder_unit_norm:
            sae.project_decoder_grad()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
        optim.step()
        if cfg.decoder_unit_norm:
            sae.renormalize_decoder()

        with torch.no_grad():
            fired_now = (f > 0).any(dim=0)
            ever_fired |= fired_now
            if steps_since_fired is not None:
                steps_since_fired += 1
                steps_since_fired[fired_now] = 0

        if step % cfg.log_every == 0:
            with torch.no_grad():
                dead_pct = 100.0 * (1.0 - ever_fired.float().mean().item())
                dead_win_pct = (100.0 * float((steps_since_fired > cfg.dead_win_steps).float().mean())
                                if steps_since_fired is not None else float("nan"))
                fq = freq.detach()
                z = sae.gate_preacts(x)
                theta = sae.theta
                entry = StepLog(
                    step=step, loss=float(loss), mse=float(mse),
                    l0_penalty=float(l0_pen),
                    hard_l0=float((z > theta).to(z.dtype).sum(dim=-1).mean()),
                    lr=lr, lambda_l0=lam, dead_pct=dead_pct,
                    freq_penalty=float(f_pen), lambda_freq=lam_f,
                    max_freq=float(fq.max()),
                    pct_dense=100.0 * float((fq > cfg.dead_freq_threshold).float().mean()),
                    theta_median=float(theta.median()), theta_min=float(theta.min()),
                    z_std=float(z.std()),
                    window_occ_pct=100.0 * float(((z - theta).abs() < (sae.bandwidth / 2)).float().mean()),
                    dead_win_pct=dead_win_pct,
                    enc_norm_mean=float(sae.W_enc.norm(dim=0).mean()),
                    dec_norm_mean=float(sae.W_dec.norm(dim=0).mean()),
                )
            log_f.write(json.dumps(asdict(entry)) + "\n")
            log_f.flush()
            log.info("step=%d loss=%.4f mse=%.4g l0=%.1f dead=%.1f%% win=%.3f%% th_med=%.3g zstd=%.3g",
                     step, entry.loss, entry.mse, entry.hard_l0, entry.dead_pct,
                     entry.window_occ_pct, entry.theta_median, entry.z_std)

            # -- health alarms (log-only): known collapse signatures ----------
            if entry.window_occ_pct < 0.05:
                if occ_low_since is None:
                    occ_low_since = step
                elif step - occ_low_since >= 5_000 and not occ_alarm:
                    occ_alarm = True
                    log.error("HEALTH: STE window occupancy < 0.05%% for %d steps — "
                              "thresholds are receiving ~no gradient.", step - occ_low_since)
            else:
                occ_low_since, occ_alarm = None, False
            zstd_hist.append((step, entry.z_std))
            while zstd_hist and zstd_hist[0][0] < step - 25_000:
                zstd_hist.pop(0)
            zstd_ref = min(v for _, v in zstd_hist)
            if entry.z_std > 2.0 * zstd_ref and math.isfinite(entry.z_std):
                if not zstd_alarm:
                    zstd_alarm = True
                    log.error("HEALTH: z_std=%.3g doubled within 25k steps (window min %.3g) — "
                              "encoder-scale runaway.", entry.z_std, zstd_ref)
            else:
                zstd_alarm = False

        if step % cfg.ckpt_every == 0 or step == cfg.n_steps:
            ckpt = _save_checkpoint(sae, optim, step, ever_fired, steps_since_fired, out_dir)
            log.info("Wrote %s", ckpt)

    if final_step < cfg.n_steps and final_step > start_step and final_step % cfg.ckpt_every != 0:
        _save_checkpoint(sae, optim, final_step, ever_fired, steps_since_fired, out_dir)

    log_f.close()
    log.info("Training done in %.1fs", time.time() - t0)
    return out_dir / f"sae_step_{final_step:07d}.safetensors"
