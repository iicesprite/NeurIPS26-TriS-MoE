<div align="center">
<h2 align="center">
<b>Structure, Subspace and System: Push the Real Limit of Extremely Low-Bit Quantization for MoE-LLMs </b>
</h2>
<div>
<a target="_blank" href="https://scholar.google.com/citations?user=lrJ0VWYAAAAJ&hl=zh-CN">Jiaqi&#160;Zhao</a><sup>1*</sup>,
<a target="_blank" href="https://openreview.net/profile?id=~Zichen_Li3">Zichen&#160;Li</a><sup>1*</sup>,
<a target="_blank" href="https://openreview.net/profile?id=~Yixuan_Dong2">Yixuan&#160;Dong</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=6EUV_UMAAAAJ&hl=en">Miao&#160;Zhang</a><sup>1&#9993</sup>,
<a target="_blank" href="https://homepage.hit.edu.cn/guanweili">Weili&#160;Guan</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=yywVMhUAAAAJ&hl=en">Liqiangi&#160;Nie</a><sup>1</sup>,
</div>
<sup>1</sup>Harbin Institute of Technology, Shenzhen&#160&#160&#160</span>
<br />
<sup>*&#160;</sup>Equal Contribution&#160;&#160;</span>
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<!-- <div align="center">
    <a href="https://aclanthology.org/2024.acl-long.122/" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
</div> -->
</div>

## Introduction

Mixture-of-Experts (MoE) large language models (LLMs) suffer significant inference and storage overheads, which makes extremely low-bit quantization (sub 2-bit) highly desirable. However, existing methods for dense LLMs typically quantize each expert independently and require it to approximate the original mapping over the entire input space. We argue that this formulation is overly conservative and structurally mismatched for MoE: 1) experts in the same MoE layer are not fully independent but operate on a shared hidden representation space. Under low-bit budgets, independently binarizing each expert will distort the activation-sensitive input geometry, which leads to severe performance degradation, and 2) due to routing-induced specialization, each expert only serves a concentrated subset of tokens, whose activations typically occupy a much narrower subspace than that of dense layers. More importantly, when the weights approach 1-bit, the dominant deployment bottleneck is no longer the weights themselves, but the metadata (e.g., scaling factors, bitmap, grouping index) which can raise the real effective inference precision to around 3-bit. This issue, however, has largely been overlooked in prior works. Building on these perspectives, we introduce a novel extremely low-bit quantization framework for MoE-LLMs, called TriS-MoE, which considers Structure, Subspace and System. Specifically, we first extract a shared high-precision input-side backbone to preserve activation-sensitive input geometry, while binarizing the remaining components to save costs. Secondly, we propose to redirect expert quantization errors into the null space of routed activations to minimize output perturbation on the subspace actually used by each expert. Finally, we co-design a bitmap compression method based on Golomb-Rice coding which approaches the Shannon entropy lower bound and a specialized streaming loading mechanism to reduce metadata overhead during inference. Extensive experiments on MoE-LLMs demonstrate our TriS-MoE outperforms the strongest baseline by 8.31% average accuracies, and also achieves an average 6.32x reduction in inference memory on real systems, enabling Qwen3.5-35B-A3B and Mixtral-8x7B to run on a single consumer-GPU.

## Contents

- [Environment setup](#environment-setup)
- [Kernel compilation](#kernel-compilation)
- [Quantization](#quantization)
- [Checkpoint export: block-GR and streaming](#checkpoint-export-block-gr-and-streaming)
- [Benchmark Evaluation](#benchmark-evaluation)
- [Acknowledgements](#Acknowledgements)
- [License](#license)



## Environment setup

Use the default environment for the OLMoE example below. For Qwen3.5, use the
separate `TriS_Qwen35` environment instead.

### Default environment

Use Python 3.12 with the `TriSMoE` Conda environment. If the environment already
exists, activate it instead of creating it again.

```bash
conda create -n TriSMoE python=3.12 pip setuptools wheel -y
conda activate TriSMoE
```

Install PyTorch from the official
[CUDA 11.8 wheel index](https://pytorch.org/get-started/previous-versions/),
then install the remaining core dependencies:

```bash
python -m pip install "torch==2.6.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install \
  "transformers==4.55.0" \
  "accelerate==1.13.0" \
  "datasets==4.8.4" \
  "numpy==2.4.3" \
  "scipy==1.17.1" \
  "safetensors==0.7.0" \
  "huggingface-hub==0.36.2" \
  "packaging==26.0" \
  "tqdm==4.67.3" \
  "pyparsing==3.3.2" \
  "sentencepiece==0.2.1" \
  "protobuf==7.34.1"
```

### Qwen3.5 environment

The following core dependency versions match the `TriS_Qwen35` environment. It
uses Python 3.12, `transformers==5.7.0`, and `huggingface-hub==1.12.0` for Qwen3.5
support. If this environment already exists, activate it instead of creating it
again.

```bash
conda create -n TriS_Qwen35 python=3.12 pip setuptools wheel -y
conda activate TriS_Qwen35

python -m pip install "torch==2.6.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install \
  "transformers==5.7.0" \
  "accelerate==1.13.0" \
  "datasets==4.8.4" \
  "numpy==2.4.3" \
  "scipy==1.17.1" \
  "safetensors==0.7.0" \
  "huggingface-hub==1.12.0" \
  "packaging==26.0" \
  "tqdm==4.67.3" \
  "pyparsing==3.3.2" \
  "sentencepiece==0.2.1" \
  "protobuf==7.34.1"
```

Both environments use PyTorch 2.6.0 with CUDA 11.8 and share the kernel
compilation instructions below.

### Repository imports (both environments)

Use the bundled, modified AutoGPTQ source rather than a separate PyPI installation:

```bash
export REPO_ROOT="$PWD"
export PYTHONPATH="$REPO_ROOT/AutoGPTQ:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Repeat these exports from the repository root when starting a new shell.

## Kernel compilation

Install a complete **CUDA 11.8 toolkit**, including `nvcc`, headers, and libraries,
and a compatible host compiler, such as GCC/G++ 11. 

Point `CUDA_HOME` to your toolkit installation. The path below is an example;
`TORCH_CUDA_ARCH_LIST=8.0` targets an NVIDIA A100 and should be adjusted for your GPU.
The build uses `g++` from `PATH`.

```bash
export CUDA_HOME=/path/to/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"

nvcc --version
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'

(
  cd "$REPO_ROOT/AutoGPTQ"
  BUILD_TRIS_ONLY=1 python setup.py build_ext --inplace
)
```

Before building, check that both `nvcc` and `torch.version.cuda` report CUDA 11.8.
`BUILD_TRIS_ONLY=1` builds only `autogptq_TriS_cuda`, skipping unrelated AutoGPTQ
extensions. Rebuild after changing the CUDA sources.

Check that the compiled extension can be imported; the current kernel ABI is `2`:

```bash
python - <<'PY'
import torch
import autogptq_TriS_cuda as kernel

print("Extension:", kernel.__file__)
print("Kernel ABI:", kernel.TRIS_COMPACT3_KERNEL_ABI_VERSION)
PY
```

## Quantization

Set `MODEL_PATH` to a local model directory or a Hugging Face model ID. This OLMoE
example uses WikiText-2 calibration, `fit_nohead` rank selection, and `vector`
correction. It keeps router weights in their original precision, evaluates
perplexity, and saves a real-quantized checkpoint for export.

```bash
MODEL_PATH="allenai/OLMoE-1B-7B-0125"
QUANT_DIR="checkpoints/olmoe-quant"

CUDA_VISIBLE_DEVICES=0 TRIS_COMPACT3_BLOCK_GR=0 \
  python tris_main.py "$MODEL_PATH" wikitext2 \
    --device cuda:0 \
    --nsamples 128 \
    --seqlen 4096 \
    --seed 0 \
    --blocksize 128 \
    --percdamp 0.01 \
    --idx_rule fit_nohead \
    --ns_rule vector \
    --quant_router false \
    --real_quant true \
    --save "$QUANT_DIR" \
    --eval
```

- `--real_quant true` packs weights into the deployment representation.
  `--save` writes the checkpoint as safetensors; the default shard limit is `5GB`,
  configurable with `--save_max_shard_size`.
- `--eval` evaluates WikiText-2 perplexity after quantization. Results and run
  configuration are written to the console and `runtime_log/`.
- `TRIS_COMPACT3_BLOCK_GR=0` leaves block-GR conversion to the export step, so the
  checkpoint can serve as the source for both output formats below.
- For **fake-quantization and PPL evaluation without saving**, change
  `--real_quant true` to `--real_quant false` and omit `--save "$QUANT_DIR"`.
  Fake-quantized floating-point weights are not an input to the export workflow.

Calibration data is cached under `cache/` by default; set `TRIS_EVAL_CACHE_DIR`
to use another location. The first run needs access to the model and dataset,
either online or through an existing local cache.

## Checkpoint export: block-GR and streaming

First build a grouped sparse checkpoint from the real-quantized checkpoint. Then
export **two separate variants from that same sparse source**:

```text
Real-quantized checkpoint
  └── Grouped sparse checkpoint
        ├── block-GR checkpoint
        └── Streaming sparse checkpoint
```

Use a different, empty output directory for each step. The initial grouped export
accepts sharded checkpoints; each export below writes a single
`grouped_runtime_payload.safetensors` file and copies the model configuration.
Keep the complete output directory when moving a checkpoint.

### Build the grouped sparse source

This step loads the quantized model onto the selected GPU and builds grouped
expert payloads. Keep block-GR and CPU staging disabled while creating the sparse
source.

```bash
GROUPED_DIR="checkpoints/olmoe-grouped-sparse"
BLOCK_GR_DIR="checkpoints/olmoe-block-gr"
STREAMING_DIR="checkpoints/olmoe-streaming"

CUDA_VISIBLE_DEVICES=0 \
TRIS_COMPACT3_BLOCK_GR=0 TRIS_GROUPED_QGMAP_CPU_STAGING=0 \
  python tools/export_grouped_sparse_checkpoint.py \
    --source-dir "$QUANT_DIR" \
    --output-dir "$GROUPED_DIR" \
    --device cuda:0
```

### Export block-GR

```bash
python tools/export_grouped_blockgr_checkpoint.py \
  --source-dir "$GROUPED_DIR" \
  --output-dir "$BLOCK_GR_DIR" \
  --blockgr-mode compressed
```

This produces block-wise Golomb-Rice maps for grouped expert projections and
non-grouped compact3 modules. By default, the exported checkpoint removes the
original sparse maps; add `--keep-sparse` only if you also need those tensors in
the block-GR output. The grouped sparse source is unchanged.

### Export streaming sparse

```bash
python tools/export_grouped_streaming_sparse_checkpoint.py \
  --source-dir "$GROUPED_DIR" \
  --output-dir "$STREAMING_DIR"
```

Use `GROUPED_DIR`, **not `BLOCK_GR_DIR`**, as the source. The streaming exporter
requires sparse maps and does not convert block-GR maps back to sparse form.
Both final export steps operate on CPU; allow enough host memory and disk space
to read and write the complete checkpoint.

### Runtime mode selection

Before loading either exported checkpoint with the bundled AutoGPTQ runtime, set:

```bash
export TRIS_GROUPED_EXPERTS=1
export TRIS_DEPLOYMENT_ONLY=1
```

Select the format-specific environment values:

| Environment variable | block-GR | Streaming sparse |
| --- | --- | --- |
| `TRIS_COMPACT3_BLOCK_GR` | `1` | `0` |
| `TRIS_GROUPED_QGMAP_CPU_STAGING` | `0` | `1` |

Streaming keeps sparse map bitmaps and prefix data in pinned CPU memory and
stages them to the GPU for execution. It does not offload all model weights or
the KV cache. Exporting the streaming checkpoint alone does not enable streaming;
set the runtime variables before model loading.

## Benchmark Evaluation

### QA EValuation

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate performance on QA datasets. 

### Model Inference Evaluation

Use `./tools/benchmark_e2e_grid.py` as an example for model loading and inferencing, we have only supported OLMoE for fast decoding yet. 


## Acknowledgements

Parts of this codebase are adapted from AutoGPTQ (https://github.com/AutoGPTQ/AutoGPTQ) and ARB-LLM (https://github.com/ZHITENGLI/ARB-LLM). We sincerely thank them for their outstanding contributions.

## License
TriS-MoE is licensed under the [Apache License 2.0](LICENSE).
Third-party code retains its original licenses; see
[AutoGPTQ/LICENSE](AutoGPTQ/LICENSE).