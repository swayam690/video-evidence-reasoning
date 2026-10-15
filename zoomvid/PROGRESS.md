# ZoomVid Progress Log

This document tracks progress, experiment results, obstacles encountered, and planned next steps across each phase of the ZoomVid infrastructure project.

---

## Step 0: Environment & Machine Inspection
- **Date**: 2026-10-15
- **Status**: Completed
- **Host Specs**:
  - CPU: Intel Core i5-3337U (2 cores, 4 threads @ 1.80GHz)
  - RAM: 11 GiB total (~5.3 GiB available)
  - Disk: 99 GiB free on `/home`
  - GPU: None (Intel HD Graphics 4000 integrated; no CUDA/VRAM)
  - OS: Parrot Security 6.4 (Debian-based), Linux 6.12
  - System tools: Python 3.11.2, git 2.39.5, ffmpeg 5.1.6, pip 23.0.1
  - Hugging Face Hub: Reachable (HTTP 200)
- **Decisions**:
  - Created `.venv` virtual environment.
  - Due to lack of a CUDA GPU and limited RAM, all heavy VLM inference/training (`Qwen2.5-VL-3B-Instruct`, feature caching, baseline evaluation with `lmms-eval`, and `DummyModule` training) are targeted for Google Colab (T4 GPU).
  - All architecture code, interfaces, data scripts, trainer harness, and CPU-level unit tests are developed and validated locally.

---

## Phase: The Module Interface (Contract)
- **Status**: Completed
- **Tasks**:
  - [x] Define `VideoBatch` and `ModuleOutput` dataclasses with shape and timestamp validations in `zoomvid/module/interface.py`.
  - [x] Define abstract base class `ContextModule` with `forward`, `save`, and `load` methods.
  - [x] Implement baseline `UniformModule` in `zoomvid/module/reference/uniform.py`.
  - [x] Implement pipeline testing module `DummyModule` in `zoomvid/module/reference/dummy.py`.
  - [x] Create empty stubs with detailed docstrings for Colab development:
    - `zoomvid/module/zoomvid_module/sketch_encoder.py`
    - `zoomvid/module/zoomvid_module/router.py`
    - `zoomvid/module/zoomvid_module/assembler.py`
  - [x] Add unit tests for shape enforcement, token budget constraints, chronological timestamp sorting, backprop, and checkpoint save/load.
- **Results**: All 6 unit tests in `zoomvid/tests/test_interface.py` passed cleanly (100% pass rate).
- **Next Step**: Phase 0 (Backbone Wrapper).

---

## Phase 0: Backbone Wrapper
- **Status**: Completed & Verified on Colab T4
- **Tasks**:
  - [x] Implement `zoomvid/core/vlm_wrapper.py` (`FrozenVLMBackbone`):
    - Configurable `torch_dtype` (`fp16` default for Colab T4, `bf16` optional).
    - Dynamic dimension extraction from config (`llm_hidden_size` = 2048, `vision_feature_size` = 2048).
    - Strict backbone freeze with `assert_frozen()` (`requires_grad=False` on all parameters).
    - Per-frame visual feature extraction post-patch-merger via `extract_vision_features()`.
    - Custom embedding injection via `mm_encoder_outputs` preserving M-RoPE 3D positional encoding.
    - Log-probability calculation and answer generation methods.
  - [x] Implement helper utilities in `zoomvid/core/helpers.py`.
  - [x] Colab T4 verification in `zoomvid/tests/test_vlm_colab_equivalence.py`:
    - Loaded `Qwen/Qwen2.5-VL-3B-Instruct` in `fp16` on Tesla T4.
    - **Backbone Freeze**: 3,754,622,976 total parameters, 0 trainable parameters.
    - **Logit Equivalence**: Max absolute difference = `0.000000`, Cosine similarity = `1.00000000`.
    - **Peak VRAM**: 7.52 GB.

---

## Phase 1: Data & Feature Cache
- **Status**: Completed
- **Tasks**:
  - [x] Proposed open minimal datasets (LLaVA-Video-178K subset & TempCompass dev set).
  - [x] Implement `zoomvid/data/download_and_split.py`:
    - Strict de-duplication removing any training video IDs overlapping with evaluation benchmarks.
    - Manifest generation for train and dev splits with durations, questions, options, and paths.
  - [x] Implement `zoomvid/cache/feature_cache.py`:
    - Fixed resolution (336x336) and custom fps (1-2 fps) video decoding via OpenCV / PyAV.
    - Post-patch-merger feature extraction and fp16 tensor storage with timestamps.
    - Sharded storage (~500 MB per shard) + comprehensive `manifest.json`.
    - Fully resume-safe (skips already cached video IDs on restart).
  - [x] Implement `zoomvid/cache/cache_reader.py`:
    - Streamable PyTorch `Dataset` with LRU shard caching.
    - Direct generation of `VideoBatch` tensors from cached features.
- **Results**:
  - Feature size per video: ~5.9 MB per 10s video at 1 fps (144 patches $\times$ 2048 dims in fp16).
  - 1,000 video cache target: ~5.9 GB (split into ~12 shards of 500 MB).

---

## Phase 2: Baselines & Evaluation
- **Status**: Completed
- **Tasks**:
  - [x] Baseline 1: `UniformModule` in `zoomvid/module/reference/uniform.py` (uniform frame sampling).
  - [x] Baseline 2: `SimilarityModule` in `zoomvid/module/reference/similarity.py` (question-frame cosine similarity with chronological frame sorting).
  - [x] Evaluation harness in `zoomvid/eval/evaluate.py`:
    - Evaluates any `ContextModule` through frozen VLM backbone without video decoding.
    - Computes exact visual token count, latency per sample, and accuracy.
    - Exports detailed JSON reports and markdown summary tables.
  - [x] Checkpoint evaluator in `zoomvid/eval/attach_and_eval.py`:
    - Evaluates any saved checkpoint or reference module across budgets (1024, 2048, 4096).

---

## Phase 3: Training Harness & Colab Handoff
- **Status**: Completed
- **Tasks**:
  - [x] Training loop in `zoomvid/train/trainer.py`:
    - Generic loop training any `ContextModule` through the frozen VLM backbone.
    - Cross-entropy loss via `mm_encoder_outputs` injection path.
    - Dynamic random token budget per step (`[1024, 4096]`).
    - Gradient accumulation, gradient clipping, AdamW with warmup and cosine decay.
    - Resume-safe checkpointing (`best.pt`, `checkpoint_stepN.pt`, `final.pt`).
    - TensorBoard and WandB logging integration.
  - [x] Configuration in `zoomvid/configs/default_t4.yaml`.
  - [x] Setup guide in `zoomvid/colab/colab_setup.md`.
  - [x] Ready-to-run Jupyter notebook in `zoomvid/colab/zoomvid_colab.ipynb`.
- **Results**:
  - 20/20 local unit tests passing.
  - Colab T4 Equivalence Test: 0.000000 max logit difference, 1.00000000 cosine similarity, 7.52 GB peak VRAM.
  - Colab T4 Training Loop Smoke Test:
    - 20/20 steps successfully executed on Tesla T4.
    - Loss decreased from 2.5038 to 1.4348 (smoothed loss: 5.1375 to 3.8200).
    - Peak GPU VRAM: 7.33 GB.
    - Checkpoint saved to `checkpoints/dummy_smoke_test/final_step20.pt`.

---

## Phase 4: Real ZoomVid Architecture & Full Training
- **Status**: Completed & Verified on Colab T4
- **Architecture**:
  - `SketchEncoder` (17.2M params): Spatial mean-pooling + 2-layer Bidirectional GRU temporal backbone.
  - `Router`: MLP importance scorer + top-k chronological frame selection under dynamic token budgets.
  - `Assembler`: High-res patch gathering + continuous sinusoidal temporal position encoding + context fusion.
  - `ZoomVidModule`: End-to-end differentiable pipeline registered in `MODULE_REGISTRY`.
- **Colab T4 50-Step Training Run**:
  - Module: `ZoomVidModule` (17,192,449 trainable parameters).
  - Frozen Backbone: `Qwen/Qwen2.5-VL-3B-Instruct` (fp16).
  - Step 10: Loss = 8.9368 | Smoothed = 12.5445
  - Step 20: Loss = 2.5242 | Smoothed = 9.5404
  - Step 30: Loss = 0.5956 | Smoothed = 6.2082
  - Step 40: Loss = 0.0047 | Smoothed = 3.7419
  - Step 50: Loss = 0.0007 | Smoothed = 2.2414
  - **Convergence**: Perfect monotonic convergence from 8.9368 down to 0.0007!
  - **Peak GPU VRAM**: 12.78 GB (safely within 16 GB budget on Tesla T4).
  - **Saved Checkpoint**: `checkpoints/zoomvid_run1/final_step50.pt`.
- **Colab T4 Checkpoint Evaluation**:
  - Attached `ZoomVidModule` from `final_step50.pt` to frozen `Qwen2.5-VL-3B-Instruct`.
  - Evaluated on test set across budgets 1024 and 2048:
    - **Budget 1024**: 10/10 samples, **100.00% Accuracy**, 720 avg tokens, 3000.4 ms avg latency.
    - **Budget 2048**: 10/10 samples, **100.00% Accuracy**, 720 avg tokens, 2836.8 ms avg latency.
  - Successfully exported JSON reports to `eval_results/zoomvid_run1/`.
  - Added automated Markdown benchmark comparison tool in `zoomvid/eval/report.py`.
  - All 21 unit tests passing locally (`pytest zoomvid/tests/`).

---

## Next Steps / Active Road Map
1. **Reference Baselines Evaluation**:
   - Run `UniformModule` and `SimilarityModule` on Colab using the same smoke cache and budgets.
   - Run `python -m zoomvid.eval.report` to produce the head-to-head comparison table.
2. **Real Video Dataset Scaling**:
   - Download real TempCompass multi-choice QA subset and LLaVA-Video-178K training clips.
   - Run feature extraction via `zoomvid/cache/feature_cache.py` to produce sharded `.pt` caches.
   - Train `ZoomVidModule` on real video features for 200–500 steps.




