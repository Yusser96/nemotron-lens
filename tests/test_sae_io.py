import json

import pytest
import torch

from nemotron_lens import SAE, SAEConfig, TrainingSAE


def _random_sae(d_in=16, d_sae=48, **cfg_kw) -> SAE:
    torch.manual_seed(0)
    sae = SAE(SAEConfig(d_in=d_in, d_sae=d_sae, **cfg_kw))
    for p in sae.parameters():
        p.data.normal_(0, 0.5)
    sae.threshold.data.abs_()
    return sae


def test_save_load_roundtrip(tmp_path):
    md = {"model_name": "m", "hook_name": "backbone.layers.2", "l0_target": 10}
    sae = _random_sae(metadata=md)
    sae.save_pretrained(tmp_path / "sae")
    loaded = SAE.load_from_disk(tmp_path / "sae")
    for k, v in sae.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k
    assert loaded.cfg.metadata == md
    assert loaded.hook_name == "backbone.layers.2"

    x = torch.randn(5, 16)
    assert torch.allclose(sae(x), loaded(x))


def test_cfg_json_matches_saelens_schema(tmp_path):
    sae = _random_sae()
    sae.save_pretrained(tmp_path / "sae")
    cfg = json.loads((tmp_path / "sae" / "cfg.json").read_text())
    for key in ("d_in", "d_sae", "dtype", "device", "apply_b_dec_to_input",
                "normalize_activations", "reshape_activations", "metadata", "architecture"):
        assert key in cfg, key
    assert cfg["architecture"] == "jumprelu"
    assert (tmp_path / "sae" / "sae_weights.safetensors").exists()


def test_encode_is_jumprelu(tmp_path):
    sae = _random_sae()
    x = torch.randn(7, 16)
    pre = x @ sae.W_enc + sae.b_enc
    expected = torch.relu(pre) * (pre > sae.threshold).float()
    assert torch.allclose(sae.encode(x), expected)
    # apply_b_dec_to_input subtracts b_dec first
    sae_b = _random_sae(apply_b_dec_to_input=True)
    pre_b = (x - sae_b.b_dec) @ sae_b.W_enc + sae_b.b_enc
    expected_b = torch.relu(pre_b) * (pre_b > sae_b.threshold).float()
    assert torch.allclose(sae_b.encode(x), expected_b)


def test_non_jumprelu_architecture_rejected():
    with pytest.raises(NotImplementedError):
        SAEConfig.from_dict({"d_in": 4, "d_sae": 8, "architecture": "topk"})


def test_saelens_cross_load(tmp_path):
    """Anything we save must load with STOCK sae_lens and compute identically."""
    sae_lens = pytest.importorskip("sae_lens")

    ours = _random_sae(metadata={"model_name": "m", "hook_name": "backbone.layers.2"})
    ours.save_pretrained(tmp_path / "sae")
    theirs = sae_lens.SAE.load_from_disk(tmp_path / "sae")
    assert type(theirs).__name__ == "JumpReLUSAE"

    x = torch.randn(9, 16)
    assert torch.allclose(ours.encode(x), theirs.encode(x), atol=1e-6)
    assert torch.allclose(ours.decode(ours.encode(x)), theirs.decode(theirs.encode(x)), atol=1e-6)


def test_training_sae_to_inference_fold_parity():
    """The folded inference SAE must reproduce the training forward on raw
    activations (same convention as nemotron-sae's parity test: mean-rel error)."""
    torch.manual_seed(0)
    tsae = TrainingSAE(d_in=16, d_sae=64)
    with torch.no_grad():
        tsae.W_enc.normal_(0, 0.5)
        tsae.W_dec.normal_(0, 0.5)
        tsae.b_enc.normal_(0, 0.1)
        tsae.b_dec.normal_(0, 0.1)
        tsae.log_theta.fill_(torch.log(torch.tensor(0.05)).item())
        tsae.norm_factor.fill_(3.7)

    inf = tsae.to_inference(metadata={"hook_name": "backbone.layers.0"})
    x = torch.randn(256, 16) / 3.7
    with torch.no_grad():
        ref, feats = tsae(x)
        got = inf(x)
    assert int((feats > 0).sum()) > 0, "fold parity check exercised no latents"
    scale = ref.abs().max().clamp_min(1e-8)
    mean_rel = ((ref - got).abs().mean() / scale).item()
    assert mean_rel == pytest.approx(0.0, abs=1e-5)
    # unit-norm decoder rows (SAEBench probe evals rely on this)
    assert torch.allclose(inf.W_dec.norm(dim=1), torch.ones(64), atol=1e-4)


def test_training_sae_raw_theta_mode():
    tsae = TrainingSAE(d_in=8, d_sae=16, threshold_param="raw", threshold_init=0.01)
    assert "theta" in dict(tsae.named_parameters())
    x = torch.randn(4, 8)
    xhat, f = tsae(x)
    assert xhat.shape == (4, 8) and f.shape == (4, 16)
