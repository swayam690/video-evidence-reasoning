# ZoomVid Google Colab Setup & Training Guide

This guide walks through running feature caching, baseline evaluation, and module training on a **Google Colab Tesla T4 GPU (16 GB VRAM)**.

---

## 1. Colab Runtime Configuration
- **Hardware Accelerator**: T4 GPU (Free tier).
- **RAM**: Standard (~12.7 GB system RAM).
- **Precision**: `fp16` (Qwen2.5-VL-3B uses ~7.52 GB VRAM in fp16).

---

## 2. Directory Structure on Google Drive
We recommend mounting Google Drive to persist feature shards and training checkpoints across sessions:
```
/content/drive/MyDrive/zoomvid/
  cache/                   # Sharded vision features (~500 MB per shard) + manifest.json
  checkpoints/             # Trained module checkpoints (.pt files)
  eval_results/            # Benchmark metrics JSON files
```

---

## 3. Step-by-Step Execution

### Step 3.1: Mount Drive & Install Dependencies
```python
from google.colab import drive
drive.mount('/content/drive')

# Create persistent folders on Drive
!mkdir -p "/content/drive/MyDrive/zoomvid/cache"
!mkdir -p "/content/drive/MyDrive/zoomvid/checkpoints"
!mkdir -p "/content/drive/MyDrive/zoomvid/eval_results"

# Clone repository & install in editable mode
%cd /content/videomodelling
!pip install -r zoomvid/requirements.txt
!pip install -e .
```

### Step 3.2: Verify Frozen VLM Equivalence
Verify that the 3B model is 100% frozen and custom injection logits match native forward:
```bash
!python zoomvid/tests/test_vlm_colab_equivalence.py
```
*Expected result: 0 trainable parameters, 7.52 GB peak VRAM, 0.000000 logit difference.*

### Step 3.3: Extract & Cache Vision Features (Run Once)
Decode videos at 1 fps (336x336) and save patch merger features in fp16 (~5.9 MB per 10s video):
```bash
!python -m zoomvid.data.download_and_split --output-dir data/ --max-train 1000 --max-eval 200
```
```bash
!python -m zoomvid.cache.feature_cache \
    --data-manifest data/train_llava_manifest.json \
    --output-dir /content/drive/MyDrive/zoomvid/cache \
    --fps 1.0 \
    --resolution 336 336 \
    --shard-size-mb 500
```

### Step 3.4: Evaluate Reference Baselines
Evaluate `UniformModule` and `SimilarityModule` across budgets (1k, 2k, 4k tokens):
```bash
# Uniform frame sampling baseline
!python -m zoomvid.eval.attach_and_eval \
    --module-type uniform \
    --cache-dir /content/drive/MyDrive/zoomvid/cache \
    --eval-manifest data/dev_tempcompass_manifest.json \
    --budgets 1024 2048 4096 \
    --output-dir /content/drive/MyDrive/zoomvid/eval_results/uniform

# Question-frame similarity baseline
!python -m zoomvid.eval.attach_and_eval \
    --module-type similarity \
    --cache-dir /content/drive/MyDrive/zoomvid/cache \
    --eval-manifest data/dev_tempcompass_manifest.json \
    --budgets 1024 2048 4096 \
    --output-dir /content/drive/MyDrive/zoomvid/eval_results/similarity
```

### Step 3.5: Train `DummyModule` Pipeline Smoke Test
Verify loss reduction, gradient accumulation, and checkpoint save/resume:
```bash
!python -m zoomvid.train.trainer \
    --config zoomvid/configs/default_t4.yaml \
    --module-type dummy \
    --cache-dir /content/drive/MyDrive/zoomvid/cache \
    --train-manifest data/train_llava_manifest.json \
    --output-dir /content/drive/MyDrive/zoomvid/checkpoints/dummy_run \
    --max-steps 100
```

### Step 3.6: Plug in Your Custom ZoomVid Module
1. Implement your architecture in `zoomvid/module/zoomvid_module/`:
   - `sketch_encoder.py`
   - `router.py`
   - `assembler.py`
2. Train your custom module:
```bash
!python -m zoomvid.train.trainer \
    --config zoomvid/configs/default_t4.yaml \
    --module-type zoomvid \
    --cache-dir /content/drive/MyDrive/zoomvid/cache \
    --train-manifest data/train_llava_manifest.json \
    --output-dir /content/drive/MyDrive/zoomvid/checkpoints/zoomvid_run1 \
    --max-steps 1000
```
3. Evaluate your trained checkpoint:
```bash
!python -m zoomvid.eval.attach_and_eval \
    --checkpoint /content/drive/MyDrive/zoomvid/checkpoints/zoomvid_run1/best_checkpoint.pt \
    --cache-dir /content/drive/MyDrive/zoomvid/cache \
    --eval-manifest data/dev_tempcompass_manifest.json \
    --budgets 1024 2048 4096 \
    --output-dir /content/drive/MyDrive/zoomvid/eval_results/zoomvid_run1
```

---

## 4. T4 Resource Guidelines
- **Frozen VLM Memory**: ~7.52 GB
- **Module Forward/Backward + Activations**: ~2.0–3.5 GB
- **Total Peak VRAM**: **~9.5–11.0 GB** (Leaves ~3.5–5 GB safety margin on 16 GB T4).
- **Recommended Hyperparameters**:
  - `batch_size`: 1 (per device)
  - `gradient_accumulation_steps`: 4 to 8
  - `token_budget`: dynamic random range `[1024, 4096]`
  - `learning_rate`: `1e-4` with cosine decay and 5% warmup

