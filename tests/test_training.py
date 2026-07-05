import json

import pytest
import torch

from nemotron_lens import SAE, ShuffleBuffer, TrainConfig, train_sae
from nemotron_lens.training.trainer import estimate_norm_factor


def _chunks(n_chunks=200, n=256, d=16, scale=5.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_chunks):
        yield torch.randn(n, d, generator=g) * scale


def test_shuffle_buffer_batches_and_exhaustion():
    buf = ShuffleBuffer(_chunks(n_chunks=4, n=100), batch_size=64)
    seen = 0
    with pytest.raises(StopIteration):
        while True:
            b = buf.next_batch()
            assert b.shape[1] == 16
            assert b.shape[0] <= 64
            seen += b.shape[0]
    assert seen == 400


def test_estimate_norm_factor_vector_convention():
    x = torch.randn(200_000, 16) * 5.0
    c = estimate_norm_factor(x, d_in=16, convention="vector")
    scaled = (x * c).pow(2).sum(-1).mean()
    assert scaled.item() == pytest.approx(1.0, rel=0.02)


def test_train_smoke_resume_and_export(tmp_path):
    cfg = TrainConfig(
        d_sae=32, l0_target=4, batch_size=128, n_steps=60, warmup_steps=5,
        l0_warmup_steps=20, ckpt_every=25, log_every=10, dead_win_steps=10,
    )
    out = tmp_path / "run"
    final = train_sae(_chunks(), d_in=16, out_dir=out, cfg=cfg, device="cpu")
    assert final.name == "sae_step_0000060.safetensors" and final.exists()

    lines = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert lines[-1]["step"] == 60
    assert lines[0]["loss"] > 0
    for key in ("hard_l0", "window_occ_pct", "theta_median", "dead_win_pct"):
        assert key in lines[0]

    # resume: extend n_steps, training continues from 60 (log appends past it)
    cfg2 = TrainConfig(**{**vars(cfg), "n_steps": 90})
    final2 = train_sae(_chunks(seed=1), d_in=16, out_dir=out, cfg=cfg2, device="cpu")
    assert final2.name == "sae_step_0000090.safetensors" and final2.exists()
    lines2 = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert lines2[-1]["step"] == 90
    assert len(lines2) > len(lines)

    # checkpoint -> TrainingSAE -> folded inference SAE -> save/load round trip
    from safetensors.torch import load_file

    from nemotron_lens import TrainingSAE

    tsae = TrainingSAE(d_in=16, d_sae=32)
    tsae.load_state_dict(load_file(str(final2)))
    inf = tsae.to_inference(metadata={"hook_name": "backbone.layers.0.mixer"})
    inf.save_pretrained(tmp_path / "exported")
    loaded = SAE.load_from_disk(tmp_path / "exported")
    x = torch.randn(8, 16)
    assert torch.allclose(inf(x), loaded(x))


def test_loss_decreases_on_learnable_data(tmp_path):
    """On low-rank data the reconstruction term must fall substantially."""

    def lowrank_chunks(n_chunks=400, n=256, d=16, rank=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        basis = torch.randn(rank, d, generator=g)
        for _ in range(n_chunks):
            yield torch.randn(n, rank, generator=g) @ basis

    cfg = TrainConfig(d_sae=32, l0_target=3, batch_size=256, n_steps=800,
                      warmup_steps=10, l0_warmup_steps=100, ckpt_every=800, log_every=20)
    out = tmp_path / "run"
    train_sae(lowrank_chunks(), d_in=16, out_dir=out, cfg=cfg, device="cpu")
    lines = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    first_mse = lines[0]["mse"]
    last_mse = sum(l["mse"] for l in lines[-3:]) / 3
    assert last_mse < first_mse * 0.5, (first_mse, last_mse)
