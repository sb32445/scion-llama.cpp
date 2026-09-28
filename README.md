# prism-ml-llama.cpp: `moe-corr-runtime` branch

This branch is the runtime for **Scion-35B-A3B** and for any other GGUF that
uses the Prism `PQ2_0` container or embedded LoRA adapters. It is the default
branch of this fork; `master` is an untouched upstream mirror and **cannot load
those files**.

Model: <https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B>
Write-up and build scripts: <https://github.com/sky-is-green/scion>

## Build

```bash
git clone -b moe-corr-runtime https://github.com/sky-is-green/prism-ml-llama.cpp
cd prism-ml-llama.cpp
./verify-container-support.sh          # must print OK before you build
rm -rf build
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-cli llama-server
```

- NVIDIA: `-DGGML_CUDA=ON`
- AMD: `-DGGML_HIP=ON` (tested on gfx1100, RX 7900 XT)
- CPU only: no flag

If you already have a checkout of this fork and the model will not load, delete
it and start from the clone above. A build of `master`, a stale `build/`
directory, or an older `llama-cli` earlier on your PATH all produce the same
error, and none of them are fixed by re-running cmake in place.

## If you see "invalid ggml type 142"

```
gguf_init_from_reader: tensor 'blk.0.ffn_down_exps.weight' has invalid ggml type 142. should be in [0, 43)
```

That is not a file problem, an OS problem or a hardware problem. It means the
binary you ran has never heard of the Prism container types:

| tree | `GGML_TYPE_COUNT` | `GGML_TYPE_PQ2_0` |
| --- | --- | --- |
| upstream `master` (also this fork's `master`) | 43 | absent |
| this branch | 144 | `= 142` |

`GGML_TYPE_COUNT` is the upper bound the error prints, so `[0, 43)` is a build of
upstream `master`. Check what you have:

```bash
grep -n "GGML_TYPE_PQ2_0" ggml/include/ggml.h   # expect: GGML_TYPE_PQ2_0 = 142
```

## What this branch adds on top of Prism's llama.cpp

Roughly 90 lines, all in the runtime:

- the `ffn_moe_out` LoRA branch target (attention-output branches can use a
  standard llama.cpp LoRA; the MoE-block placement needs this virtual tensor)
- embedded adapters: `adapter.embedded=true` lets a correction sidecar travel
  inside the model file, so a corrected model loads with no `--lora` step
- LoRA hooks for the `olmoe` and `qwen35moe` graphs
- import shim for the legacy `Q1_0_g128` container (type id 43), byte-identical
  to `PQ2_0`, remapped at load time so old files need no repack

The container, the quantization rule and the kernels are Prism ML's work, not
mine. See `ggml/include/ggml.h` for the type ids and `ggml/src/gguf.cpp` for the
legacy import.

## Container types in the Scion-35B-A3B release

The file is a mix, which is normal:

| type id | name | what |
| --- | --- | --- |
| 142 | `PQ2_0` | ternary expert banks (2-bit codes + one fp16 scale per 128) |
| 43 | legacy `Q1_0_g128` | the embedded rank-512 corrections, imported as `PQ2_0` |
| 8 | `Q8_0` | attention, embeddings, LM head |
| 0 | `F32` | norms, routers, output |

## Credits and license

MIT, following upstream: [llama.cpp](https://github.com/ggml-org/llama.cpp) by
the ggml-org and [Prism ML](https://github.com/PrismML-Eng/llama.cpp), with the
`Q1_0_g128` container conventions from
[CodeMasterCody3D](https://github.com/CodeMasterCody3D/prism-ml-llama.cpp).
Base weights for the release are
[empero-ai/Qwen3.8-35B-A3B-Distill](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill)
and [Qwen](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). This fork is not
affiliated with or endorsed by any of them.
