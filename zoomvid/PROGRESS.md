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
- **Status**: Completed
- **Tasks**:
  - [x] Implement `zoomvid/core/vlm_wrapper.py` (`FrozenVLMBackbone`):
    - Configurable `torch_dtype` (`fp16` default for Colab T4, `bf16` optional).
    - Dynamic dimension extraction from config (`llm_hidden_size` = 2048, `vision_feature_size` = 2048).
    - Strict backbone freeze with `assert_frozen()` (`requires_grad=False` on all parameters).
    - Methods for per-frame vision feature extraction post-patch-merger.
    - Custom embedding injection path via `inputs_embeds` (combining prefix text, module tokens, suffix text, target answer).
    - Log-probability calculation and answer generation methods.
  - [x] Implement helper utilities in `zoomvid/core/helpers.py` (timestamp tagging, budget calculation, chronological checks).
  - [x] Add local test suite in `zoomvid/tests/test_vlm_wrapper.py`:
    - Dtype parsing.
    - Token budget and timestamp formatters.
    - Parameter freezing assertion check.
    - Injection sequence assembly and loss mask formatting (-100 for prompt).
    - Numerical equivalence of injection path.
  - [x] Provide standalone Colab verification script in `zoomvid/tests/test_vlm_colab_equivalence.py`:
    - Full `Qwen/Qwen2.5-VL-3B-Instruct` fp16 equivalence test between native pipeline and custom injection path on Colab GPU.
- **Results**: All 11 unit tests in `zoomvid/tests/test_interface.py` and `zoomvid/tests/test_vlm_wrapper.py` passed cleanly (100% pass rate).
- **Next Step**: Phase 1 (Data & Feature Cache).

---

## Phase 1: Data & Feature Cache
- **Status**: Completed
- **Tasks**:
  - [x] Proposed open minimal datasets (LLaVA-Video-178K subset & TempCompass dev set) and calculated storage budgets.
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
  - [x] Comprehensive unit tests in `zoomvid/tests/test_data_and_cache.py`:
    - Normalized video ID parsing.
    - Strict de-duplication assertions.
    - Shard limit rollover and resume safety verification.
    - Cache reader streaming and `ContextModule` integration.
- **Results**:
  - Feature size per video: ~5.9 MB per 10s video at 1 fps (144 patches $\times$ 2048 dims in fp16).
  - All 15 unit tests across the test suite passed cleanly (100% pass rate).
- **Next Step**: Phase 2 (Baselines & Evaluation).

---

## Phase 2: Baselines & Evaluation
- **Status**: Pending
- **Tasks**:
  - Evaluate UniformModule (1k, 2k, 4k budgets) on TempCompass dev set.
  - Implement and evaluate Question-Frame similarity baseline (CLIP / SigLIP).

---

## Phase 3: Training Harness & Colab Handoff
- **Status**: Pending
- **Tasks**:
  - Implement `trainer.py` with random token budgets and cross-entropy loss.
  - DummyModule smoke training test.
  - Package Colab setup guide and notebook.

