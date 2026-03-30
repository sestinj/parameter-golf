# Doc-Mask Ablation Implementation (Ternary U-Net)

This folder tracks our doc-boundary masking ablation built on top of:

- `2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon`

The goal is to test whether disallowing cross-document attention improves legal leaderboard BPB, while keeping evaluation token coverage correct.

## What Was Implemented

All implementation changes are in:

- `train_gpt_cuda_ternary.py`

### 1) Config Surface (env flags)

Added flags:

- `DOC_MASK_TRAIN` - enable masking in training forward pass
- `DOC_MASK_EVAL` - enable masking in validation forward pass
- `DOC_BOUNDARY_ID` - boundary token ID (`-1` means auto-resolve from tokenizer BOS)
- `DOC_MASK_ATTN_BACKEND` - `auto|varlen|sdpa`
- `DOC_MASK_EVAL_LOG_COVERAGE` - log expected/scored token coverage
- `DOC_MASK_EVAL_STRICT_COVERAGE` - assert no missing scored tokens
- `DOC_MASK_DISABLE_SEQ_MIX` - disable non-attention sequence mixing while masked
- `DOC_MASK_COMPILE` - allow `torch.compile` even when doc mask is active
- `DOC_MASK_COMPILE_DYNAMIC` - compile with `dynamic=True` for masked mode

### 2) Attention-Level Doc Masking

Implemented segment-aware masking in attention:

- Segment IDs from boundary tokens (`cumsum` on boundary indicator)
- Segment-local RoPE positions (position resets at each document segment)
- Two masked backends:
  - `flash_attn_varlen_func` (preferred)
  - SDPA explicit mask fallback (when varlen unavailable)

### 3) Legal Evaluation Coverage

Reworked eval so masking does not silently drop scored tokens:

- No start filtering for doc masking
- Full expected token count computed deterministically
- Coverage log emitted (`expected_tokens`, `scored_tokens`)
- Strict mismatch guard raises when scored coverage is too low

This keeps eval comparable and legal under challenge constraints.

### 4) Leakage Controls for Non-Attention Mixers

When `use_doc_mask` and `DOC_MASK_DISABLE_SEQ_MIX=1`:

- `SmearModule` disabled
- `CausalConvRefiner` disabled
- MTP auxiliary heads disabled

This avoids cross-doc leakage outside attention.

### 5) Compile + Throughput Work

Original masked implementation had severe slowdown from Python/CPU packing.

Fixes applied:

- Removed CPU transfer and Python segment loops in varlen packing
- Vectorized segment boundary extraction on GPU
- Added compile path for masked mode:
  - `DOC_MASK_COMPILE=1`
  - `DOC_MASK_COMPILE_DYNAMIC=1`
- Precomputed segment metadata once per `GPT.forward` and reused per block:
  - `segment_pos_ids`
  - `segment_cu_seqlens`

## Throughput Isolation Summary

Controlled short runs (same model/hparams, 20 steps, warmup 3):

- Baseline (`off/off`, compile on): `~352 ms/step`
- Masked (early buggy path): `~8152 ms/step`
- Masked after GPU packing fix, compile off: `~882 ms/step`
- Masked with compile forced: `~483 ms/step`
- Masked with compile + forward-level precompute: `~390 ms/step`

Final residual overhead vs baseline is about `1.11x`, which is consistent with varlen path overhead.

## Key Ablation Results Observed

### Step-matched (40 steps)

- `off/off`: `2.4732` (roundtrip BPB)
- `off/on`: `2.4777`
- `on/off`: `2.4506` (from earlier 40-step run)
- `on/on`: `2.4646` (from earlier 40-step run)

### Long wallclock re-run (train-mask only, optimized path)

- `DOC_MASK_TRAIN=1`, `DOC_MASK_EVAL=0`, `MAX_WALLCLOCK_SECONDS=300`
- reached `step 770` in ~302s
- final roundtrip: `val_bpb=1.4684`
- coverage check: `expected_tokens=199680`, `scored_tokens=199680`

## Reproduction Commands

Run from this folder:

- `records/track_10min_16mb/2026-03-29_74M_Ternary_UNet_FP8_DocMask_Ablation`

### Train-mask only (optimized)

```bash
RUN_ID=ablation_trainmask_only \
OMP_NUM_THREADS=1 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=768 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=4 EMBED_DIM=254 ACTIVATION=relu2 FP_STORAGE=FP8 \
TRAIN_BATCH_TOKENS=262144 ITERATIONS=100000 WARMUP_STEPS=3 \
MAX_WALLCLOCK_SECONDS=300 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=400 \
VAL_MAX_TOKENS=200000 SLIDING_EVAL=0 \
DOC_MASK_TRAIN=1 DOC_MASK_EVAL=0 DOC_MASK_ATTN_BACKEND=auto \
DOC_MASK_COMPILE=1 DOC_MASK_COMPILE_DYNAMIC=1 \
DOC_MASK_EVAL_STRICT_COVERAGE=1 DOC_MASK_EVAL_LOG_COVERAGE=1 \
python train_gpt_cuda_ternary.py
```

### Baseline off/off

```bash
DOC_MASK_TRAIN=0 DOC_MASK_EVAL=0 python train_gpt_cuda_ternary.py
```

### Eval-mask only off/on

```bash
DOC_MASK_TRAIN=0 DOC_MASK_EVAL=1 DOC_MASK_COMPILE=1 DOC_MASK_COMPILE_DYNAMIC=1 \
python train_gpt_cuda_ternary.py
```

### Train+eval mask on/on

```bash
DOC_MASK_TRAIN=1 DOC_MASK_EVAL=1 DOC_MASK_COMPILE=1 DOC_MASK_COMPILE_DYNAMIC=1 \
python train_gpt_cuda_ternary.py
```

## Legal/Protocol Notes

- Doc-aligned **training** sampling is legal as a sampler choice.
- For **evaluation**, legality requires exact scored-token coverage and no leakage from unscored future tokens.
- Current implementation enforces coverage logging and optional strict checks to prevent accidental score inflation/deflation.

## Current Status

- Doc-mask train/eval toggles are fully implemented and active.
- Throughput bottleneck was isolated and substantially fixed.
- Remaining work (optional): rerun a fresh full 2x2 matrix at 300s using the optimized compile-enabled masked path for final ranking conclusions.
