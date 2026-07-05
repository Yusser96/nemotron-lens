# nemotron-lens

TransformerLens-style interpretability tooling for **NVIDIA Nemotron hybrid models**
(Mamba-2 + GQA-Attention + MoE) — hook any location, cache activations, train
JumpReLU SAEs, splice them into the forward pass, and evaluate them with
SAE-Bench / SAE-Lens.

TransformerLens does not support hybrid Mamba-attention architectures, and
Nemotron-3-Nano loads through `trust_remote_code`. nemotron-lens hooks the plain
HuggingFace model directly (zero-overhead PyTorch forward hooks, module paths
verified by runtime topology introspection) and keeps every SAE artifact in
**SAELens's native on-disk format**, so the SAEs you train or download here also
load with stock `sae_lens` — same repos, same folders, no conversion.

```python
from nemotron_lens import HookedNemotron, SAE

model = HookedNemotron.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

# 1. list hookable locations (canonical names; raw HF module paths work too)
model.hook_points(kinds=["resid_post", "mamba_out", "moe_out"], layers=[2, 25])
# ['blocks.2.hook_resid_post', 'blocks.25.hook_mamba_out', ...]

# 2. run with cache
logits, cache = model.run_with_cache(
    "The Eiffel Tower is in", names=["blocks.25.hook_mamba_out"]
)
acts = cache["blocks.25.hook_mamba_out"]        # (batch, seq, d_model)

# 3. load a pretrained SAE (either library reads the same HF repo!)
sae = SAE.from_pretrained("Yusser/nemotron-3-nano-30b-a3b-saes",
                          "L25_mamba_out/w16384_l0_10", device="cuda")
feats = sae.encode(acts.float().flatten(0, 1))  # (tokens, 16384), ~10 active each

# 4. splice the SAE into the forward pass (uses sae.metadata['hook_name'])
recon_logits = model.run_with_saes("The Eiffel Tower is in", sae)
```

The same SAE via stock [SAELens](https://github.com/decoderesearch/SAELens):

```python
from sae_lens import SAE  # returns a sae_lens JumpReLUSAE — identical weights
sae = SAE.from_pretrained("Yusser/nemotron-3-nano-30b-a3b-saes",
                          "L25_mamba_out/w16384_l0_10")
```

## Install

```bash
pip install nemotron-lens              # SAE loading/training, hooking API
pip install "nemotron-lens[model]"     # + mamba-ssm/causal-conv1d (run the LM; needs CUDA + nvcc)
pip install "nemotron-lens[eval]"      # + sae-lens/transformer-lens/sae-bench (eval harness)
```

The base install has no TransformerLens/SAELens dependency: loading SAEs and
training on pre-extracted activations works anywhere PyTorch does. Running the
30B model itself requires the Mamba-2 CUDA kernels (`[model]` extra) — for
compile flags on Blackwell GPUs see
[nemotron-sae](https://github.com/Yusser96/nemotron-sae)'s `setup_venv.sh`.

## Hook points

A Nemotron layer is Mamba-2 **or** attention **or** MoE. Canonical names follow
TransformerLens conventions, resolved against the model's runtime topology
(`model.save_topology(path)` dumps it):

| name | activation |
|---|---|
| `blocks.{L}.hook_resid_pre` / `hook_resid_post` | residual stream (block input / output) |
| `blocks.{L}.hook_mamba_out` | Mamba-2 mixer output |
| `blocks.{L}.hook_attn_out_prelinear` | attention mixer output |
| `blocks.{L}.hook_moe_out` | MoE block output |
| `blocks.{L}.hook_expert.{i}` / `hook_shared_expert` | routed / shared expert output |

Every API also accepts raw HF module paths (`backbone.layers.25.mixer`) — which
is exactly what published SAEs record in `metadata["hook_name"]`.

`run_with_hooks` lets you observe or **edit** any of these:

```python
def boost(acts, hook_name):
    return acts * 1.1                       # return None to observe only

logits = model.run_with_hooks(prompt, [("blocks.25.hook_mamba_out", boost)])
```

`run_with_cache(..., stop_after_cache=True)` aborts the forward right after the
deepest requested hook — activation extraction on a 30B model without paying for
the upper layers or the LM head.

## Train an SAE

The trainer is the Gemma Scope 2 JumpReLU recipe with the exact configuration
first validated on Nemotron-3-Nano (see the
[nemotron-sae](https://github.com/Yusser96/nemotron-sae) technical report):
quadratic L0-target penalty, `paper_v3` straight-through estimators, log-space
thresholds, unit-norm decoder columns with gradient projection, health telemetry
that catches the known JumpReLU collapse modes live.

```python
from nemotron_lens import ShuffleBuffer, TrainConfig, train_sae, TrainingSAE

texts = (row["text"] for row in my_hf_dataset)          # any str iterator
stream = model.stream_activations(texts, "blocks.25.hook_mamba_out",
                                  batch_size=8, context_size=2048)
buffer = ShuffleBuffer(stream, batch_size=4096)

ckpt = train_sae(buffer, d_in=model.d_model, out_dir="runs/L25_mamba",
                 cfg=TrainConfig(d_sae=16384, l0_target=10, n_steps=200_000))

# fold norm-factor + pre-encoder bias -> standard JumpReLU on raw activations
from safetensors.torch import load_file
tsae = TrainingSAE(d_in=model.d_model, d_sae=16384)
tsae.load_state_dict(load_file(ckpt))
sae = tsae.to_inference(metadata={
    "model_name": model.model.config._name_or_path,
    "hook_name": model.module_path("blocks.25.hook_mamba_out"),
})
sae.save_pretrained("export/L25_mamba_out/w16384_l0_10")   # sae_lens-native format
```

Training resumes automatically from the latest checkpoint in `out_dir`;
`train_log.jsonl` records loss/L0/dead-% plus threshold-health diagnostics
(STE window occupancy, `z_std`, θ stats) every `log_every` steps.

To publish, upload the exported folders to the HuggingFace Hub (e.g. with
`sae_lens.upload_saes_to_huggingface`) and both libraries can
`from_pretrained` them.

## Evaluate (SAE-Bench / SAE-Lens)

Both suites are TransformerLens-based; nemotron-lens bridges the gap with
SAE-Lens's `HookedProxyLM` plus the compatibility shims proven in nemotron-sae's
external evaluation harness (`pip install "nemotron-lens[eval]"`):

```python
from nemotron_lens.evals import run_saebench, run_saelens_evals

run_saelens_evals(sae, model)                       # L0, FVU, CE-loss-recovered, KL
run_saebench(sae, model, evals=["core", "sparse_probing", "scr", "tpp", "absorption"])
```

## Pretrained SAEs

13 JumpReLU SAEs (width 16384, L0*=10, 1M steps ≈ 4.1B token-activations on
Nemotron-CC-v2.1) across residual / Mamba / attention / MoE sites of
Nemotron-3-Nano-30B-A3B — to our knowledge the first public SAEs for a hybrid
Mamba-attention LLM:
[`Yusser/nemotron-3-nano-30b-a3b-saes`](https://huggingface.co/Yusser/nemotron-3-nano-30b-a3b-saes)
(per-SAE eval metrics in each `cfg.json` and on the model card).

## Related projects

- [nemotron-sae](https://github.com/Yusser96/nemotron-sae) — the full training
  pipeline these SAEs came from (sharded activation caches, sweep launcher,
  slurm scripts, technical report).
- [SAELens](https://github.com/decoderesearch/SAELens) — SAE standard library;
  nemotron-lens uses its on-disk format and (optionally) its eval stack.
- [SAEBench](https://github.com/adamkarvonen/SAEBench) — standardized SAE
  evaluation suite, driven here through the proxy bridge.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — the
  API conventions `HookedNemotron` follows.

## Citation

If you use nemotron-lens or the pretrained SAEs, please cite this repository and
Gemma Scope (arXiv:2408.05147), whose training recipe the SAEs follow.

## License

MIT. The Nemotron base model is subject to the NVIDIA Open Model License.
