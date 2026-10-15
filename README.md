# ZoomVid: Trainable Video-Context Module Infrastructure

ZoomVid is an infrastructure and research harness for training a pluggable video-context module on top of a **frozen** Vision-Language Model backbone (`Qwen/Qwen2.5-VL-3B-Instruct`).

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                              Frozen VLM                                 |
|  +---------------------+                       +---------------------+  |
|  | Vision Encoder      |                       | Language Model (LLM)|  |
|  | (patch merger out)  |                       | (inputs_embeds)     |  |
|  +----------+----------+                       +----------^----------+  |
+-------------|---------------------------------------------|-------------+
              | [B, N, P, D]                                | [B, T, D_llm]
              v                                             |
   +--------------------------------------------------------+---------+
   |                       Pluggable Module                           |
   |                                                                  |
   |   VideoBatch: frame_feats, timestamps, question_emb, budget      |
   |   --> [Sketch Encoder] -> [Router] -> [Assembler]                |
   |   --> ModuleOutput: tokens [B, T, D_llm], segment_spans, aux     |
   +------------------------------------------------------------------+
```

## Project Structure

```
zoomvid/
  configs/         # Experiment configurations (YAML)
  core/            # Backbone wrapper (vlm_wrapper.py) & token/timestamp utilities
  data/            # Dataset downloaders, deduplication & splitting scripts
  cache/           # Feature caching pipeline & shard reader/writer
  module/          # Module interface, reference baselines, and zoomvid stubs
    interface.py   # The contract: VideoBatch, ModuleOutput, ContextModule
    reference/     # UniformModule, DummyModule, SimilarityModule
    zoomvid_module/# sketch_encoder.py, router.py, assembler.py
  train/           # trainer.py: modular training harness with frozen VLM
  eval/            # Evaluation scripts & lmms-eval benchmarks
  colab/           # Google Colab setup guide and notebook template
  tests/           # Shape tests, budget enforcement, timestamp ordering
  README.md
  requirements.txt
  PROGRESS.md
```

## Module Contract

Every module subclassing `ContextModule` accepts a `VideoBatch` and produces a `ModuleOutput`:

```python
from zoomvid.module.interface import VideoBatch, ModuleOutput, ContextModule

class CustomModule(ContextModule):
    def forward(self, batch: VideoBatch) -> ModuleOutput:
        # batch.frame_feats: [B, N, P, D]
        # batch.timestamps:  [B, N] (seconds)
        # batch.token_budget: int (max T)
        # return tokens:     [B, T, D_llm]
        ...
```

