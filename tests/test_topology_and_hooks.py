import torch

from nemotron_lens import enumerate_hooks


def test_enumerate_hooks_classifies_all_kinds(tiny_model):
    sites = enumerate_hooks(tiny_model)
    kinds = {(s.layer, s.component_kind) for s in sites}
    assert (0, "mamba_out") in kinds
    assert (1, "attn_out_prelinear") in kinds
    assert (2, "moe_out") in kinds
    assert (2, "expert") in kinds and (2, "shared_expert") in kinds
    for L in range(3):
        assert (L, "resid_pre") in kinds and (L, "resid_post") in kinds
    experts = [s for s in sites if s.component_kind == "expert"]
    assert len(experts) == 4
    assert experts[0].module_path.startswith("backbone.layers.2.mixer.experts.")


def test_hook_points_filtering(hooked):
    all_pts = hooked.hook_points()
    assert "blocks.0.hook_mamba_out" in all_pts
    assert "blocks.2.hook_moe_out" in all_pts
    assert not any("expert" in p for p in all_pts)  # hidden by default
    with_experts = hooked.hook_points(include_experts=True)
    assert "blocks.2.hook_expert.3" in with_experts
    only_moe = hooked.hook_points(kinds=["moe_out"])
    assert only_moe == ["blocks.2.hook_moe_out"]
    layer1 = hooked.hook_points(layers=[1])
    assert all(p.startswith("blocks.1.") for p in layer1)


def test_module_path_resolution(hooked):
    assert hooked.module_path("blocks.0.hook_mamba_out") == "backbone.layers.0.mixer"
    assert hooked.module_path("blocks.1.hook_resid_post") == "backbone.layers.1"
    # raw module paths pass through
    assert hooked.module_path("backbone.layers.2.mixer") == "backbone.layers.2.mixer"


def test_run_with_cache_shapes_and_names(hooked, tokens):
    names = ["blocks.0.hook_mamba_out", "blocks.2.hook_moe_out",
             "blocks.1.hook_resid_post", "backbone.layers.1.mixer"]
    logits, cache = hooked.run_with_cache(tokens, names)
    assert logits.shape == (2, 7, 50)
    for n in names:
        assert cache[n].shape == (2, 7, 16), n


def test_resid_pre_equals_previous_resid_post(hooked, tokens):
    _, cache = hooked.run_with_cache(
        tokens, ["blocks.0.hook_resid_post", "blocks.1.hook_resid_pre"]
    )
    assert torch.equal(cache["blocks.0.hook_resid_post"], cache["blocks.1.hook_resid_pre"])


def test_stop_after_cache_skips_rest_of_forward(hooked, tokens):
    logits, cache = hooked.run_with_cache(
        tokens, ["blocks.0.hook_mamba_out"], stop_after_cache=True
    )
    assert logits is None
    assert cache["blocks.0.hook_mamba_out"].shape == (2, 7, 16)


def test_run_with_hooks_edits_activations(hooked, tokens):
    base = hooked(tokens)

    def zero_it(acts, name):
        return torch.zeros_like(acts)

    edited = hooked.run_with_hooks(tokens, [("blocks.0.hook_mamba_out", zero_it)])
    assert not torch.allclose(base, edited)
    # observe-only hooks (returning None) must not change the output
    seen = {}
    observed = hooked.run_with_hooks(
        tokens, [("blocks.0.hook_mamba_out", lambda a, n: seen.update(x=a.clone()))]
    )
    assert torch.allclose(base, observed)
    assert seen["x"].shape == (2, 7, 16)


def test_run_with_saes_zero_mode_matches_manual_zeroing(hooked, tokens):
    from nemotron_lens import SAE, SAEConfig

    sae = SAE(SAEConfig(d_in=16, d_sae=8,
                        metadata={"hook_name": "backbone.layers.0.mixer"}))
    via_sae = hooked.run_with_saes(tokens, sae, mode="zero")
    manual = hooked.run_with_hooks(
        tokens, [("backbone.layers.0.mixer", lambda a, n: torch.zeros_like(a))]
    )
    assert torch.allclose(via_sae, manual)


def test_run_with_saes_uses_metadata_hook_and_changes_output(hooked, tokens):
    from nemotron_lens import SAE, SAEConfig

    torch.manual_seed(2)
    cfg = SAEConfig(d_in=16, d_sae=64, metadata={"hook_name": "backbone.layers.0.mixer"})
    sae = SAE(cfg)
    for p in sae.parameters():
        p.data.normal_(0, 0.3)
    base = hooked(tokens)
    spliced = hooked.run_with_saes(tokens, sae)
    assert spliced.shape == base.shape
    assert not torch.allclose(base, spliced)
