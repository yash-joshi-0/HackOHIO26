# GLM-4.7-Flash shard experiment

The full GLM-4.7-Flash Q4_K_M model generated two coherent sentences across the M1 Air and M2 Pro over Iroh at 10.2786 decode steps/second in one trial (512-token context, 57 output steps including EOG). Source model size: 18,312,339,808 bytes, larger than either machine's 16 GiB physical RAM. Layers 0:23 ran on M1; layers 23:47 ran on M2. The connection began through a relay and the stream closed on a direct path. This is an observed result, not a matched comparison with the earlier dense Qwen 32B relay run.

Source: `unsloth/GLM-4.7-Flash-GGUF`, revision `0d32489ecb9db6d2a4fc93bd27ef01519f95474d`, SHA-256 `29837ed2c0fc5f51981adf8ac8083fcf80743c598381f13e9f06cbad0498b174`. This MoE model stores roughly 30B parameters but activates roughly 3B per token. The private patch extends the shard graph to deepseek2 MLA/MoE, retaining global layer indices and raw hidden-state boundaries. Other MoE architectures are not thereby supported.

Raw results, exported-artifact manifests, leader transport log, and runtime patch are in `results/glm-shard-2026-10-24/`. Apply the patch to private llama.cpp base `71255fefc092b0344f2acddb891388814ae961be`; rebuild all callers and libraries together because per-model `shard_begin`/`shard_end` fields change the model-params ABI. `export_prefix.py` creates each layer range without dequantizing weights. The execution protocol remains v1 as documented in `SHARD_RUNNER.md`.

Observed initialized GGML allocations were 8,762,505,728 and 9,674,387,456 bytes. These are not total process peaks or hard memory limits; memory compression was present. This experiment does not prove swap-free execution, complete device contribution accounting, or deployment readiness.

`glm_fixture.py` extracts a deliberately truncated two-layer numerical fixture (one dense layer and one MoE layer). CPU and Metal split/reference comparisons matched exactly across 16 positions. The same fixture validates physical iPad and Android native execution, described in `../mobile-probe/README.md`; its generated text is not a model-quality result.
