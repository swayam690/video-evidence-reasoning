"""Generic training loop for any ContextModule through a frozen VLM.

Supports:
- Cross-entropy loss through the frozen VLM backbone
- Random token budget per step
- Gradient accumulation
- Checkpoint saving and resuming
- WandB or TensorBoard logging
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from zoomvid.cache.cache_reader import FeatureCacheReader
from zoomvid.core.vlm_wrapper import FrozenVLMBackbone
from zoomvid.module.interface import ContextModule, VideoBatch
from zoomvid.module.reference.dummy import DummyModule
from zoomvid.module.reference.uniform import UniformModule
from zoomvid.module.reference.similarity import SimilarityModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODULE_REGISTRY = {
    "dummy": DummyModule,
    "uniform": UniformModule,
    "similarity": SimilarityModule,
}


class TrainingDataset(Dataset):
    """Training dataset that loads cached features and paired QA annotations."""

    def __init__(
        self,
        cache_reader: FeatureCacheReader,
        manifest: List[Dict[str, Any]],
    ) -> None:
        self.cache_reader = cache_reader
        # Filter manifest to only include videos present in cache
        self.samples = [
            s for s in manifest if s.get("video_id", "") in cache_reader.manifest
        ]
        logger.info("TrainingDataset: %d samples (of %d manifest entries) have cached features.",
                     len(self.samples), len(manifest))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        video_id = sample["video_id"]
        feats, times = self.cache_reader.get_features(video_id)

        # Extract question and answer from conversations or direct fields
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        if not question and "conversations" in sample:
            convs = sample["conversations"]
            for c in convs:
                if c.get("from") == "human":
                    question = c.get("value", "")
                elif c.get("from") == "gpt":
                    answer = c.get("value", "")

        return {
            "video_id": video_id,
            "frame_feats": feats,      # [N, P, D]
            "timestamps": times,        # [N]
            "question": question,
            "answer": answer,
        }


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple collation: process one sample at a time (batch size handled by accumulation)."""
    # For simplicity, return the first item (effective batch_size=1 per forward,
    # use gradient_accumulation_steps for larger effective batch sizes)
    return batch[0]


class Trainer:
    """Generic training loop for any ContextModule through a frozen VLM."""

    def __init__(
        self,
        module: ContextModule,
        backbone: FrozenVLMBackbone,
        train_dataset: TrainingDataset,
        output_dir: str = "checkpoints",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
        budget_range: Tuple[int, int] = (1024, 4096),
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 200,
        warmup_ratio: float = 0.05,
        seed: int = 42,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        prefix_text: str = "<|im_start|>user\n<|vision_start|>",
        suffix_template: str = "<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n",
    ) -> None:
        self.module = module
        self.backbone = backbone
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.grad_accum = gradient_accumulation_steps
        self.budget_range = budget_range
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        self.prefix_text = prefix_text
        self.suffix_template = suffix_template

        # Setup
        self._set_seed(seed)

        self.device = backbone.device
        self.module.to(self.device)

        # Optimizer (only module parameters — backbone is frozen)
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler with warmup
        warmup_steps = max(1, int(self.max_steps * self.warmup_ratio))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps)
            * max(0.0, 1.0 - (step - warmup_steps) / max(1, self.max_steps - warmup_steps)),
        )

        # DataLoader
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_single,
            num_workers=0,
        )

        # Logging
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
            except ImportError:
                logger.warning("TensorBoard not available. Skipping.")
                self.use_tensorboard = False

        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="zoomvid", dir=str(self.output_dir))
            except ImportError:
                logger.warning("WandB not available. Skipping.")
                self.use_wandb = False

        # Training state
        self.global_step = 0
        self.running_loss = 0.0
        self.best_loss = float("inf")
        self.train_log: List[Dict[str, Any]] = []

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save_checkpoint(self, name: str = "checkpoint") -> Path:
        """Save module checkpoint + training state."""
        ckpt_path = self.output_dir / f"{name}_step{self.global_step}.pt"
        payload = {
            "module_class": self.module.__class__.__name__,
            "in_features": self.module.in_features,
            "out_features": self.module.out_features,
            "state_dict": self.module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "seed": self.seed,
        }
        torch.save(payload, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.module.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step = ckpt["global_step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        logger.info("Resumed from step %d (best_loss=%.4f)", self.global_step, self.best_loss)

    def train_step(self, sample: Dict[str, Any]) -> float:
        """Single training step: module forward → VLM loss → backward.

        Returns the loss value (float).
        """
        # Random token budget per step
        budget = random.randint(*self.budget_range)

        frame_feats = sample["frame_feats"].unsqueeze(0).to(self.device)  # [1, N, P, D]
        timestamps = sample["timestamps"].unsqueeze(0).to(self.device)    # [1, N]
        question = sample["question"]
        answer = sample["answer"]

        batch = VideoBatch(
            frame_feats=frame_feats,
            timestamps=timestamps,
            mode="qa",
            token_budget=budget,
        )

        # Forward through trainable module
        output = self.module(batch)
        module_tokens = output.tokens  # [1, T, D_llm]
        t = module_tokens.shape[1]

        suffix = self.suffix_template.format(question=question)
        image_grid_thw = torch.tensor([[t, 1, 1]], dtype=torch.long, device=self.device)

        # Forward through frozen VLM with loss
        vlm_output = self.backbone.forward_with_injected_tokens(
            module_tokens=module_tokens,
            prefix_text=self.prefix_text,
            suffix_text=suffix,
            target_text=answer,
            image_grid_thw=image_grid_thw,
        )

        loss = vlm_output.loss
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.grad_accum
        scaled_loss.backward()

        return loss.item()

    def train(self) -> None:
        """Main training loop."""
        self.module.train()
        self.backbone.assert_frozen()

        logger.info("Starting training for %d steps...", self.max_steps)
        logger.info("Module: %s | Params: %d",
                     self.module.__class__.__name__,
                     sum(p.numel() for p in self.module.parameters() if p.requires_grad))

        data_iter = iter(self.dataloader)
        accum_loss = 0.0

        while self.global_step < self.max_steps:
            self.optimizer.zero_grad()

            step_losses = []
            for accum_idx in range(self.grad_accum):
                try:
                    sample = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    sample = next(data_iter)

                try:
                    loss_val = self.train_step(sample)
                    step_losses.append(loss_val)
                except Exception as e:
                    logger.error("Error in train_step (video=%s): %s",
                                 sample.get("video_id", "?"), e)
                    continue

            if step_losses:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1

                avg_loss = sum(step_losses) / len(step_losses)
                self.running_loss = 0.95 * self.running_loss + 0.05 * avg_loss if self.running_loss > 0 else avg_loss

                # Logging
                if self.global_step % self.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    log_entry = {
                        "step": self.global_step,
                        "loss": round(avg_loss, 4),
                        "smoothed_loss": round(self.running_loss, 4),
                        "lr": lr,
                    }
                    self.train_log.append(log_entry)

                    logger.info(
                        "Step %d/%d | Loss: %.4f | Smoothed: %.4f | LR: %.2e",
                        self.global_step, self.max_steps, avg_loss, self.running_loss, lr,
                    )

                    if self.tb_writer:
                        self.tb_writer.add_scalar("train/loss", avg_loss, self.global_step)
                        self.tb_writer.add_scalar("train/smoothed_loss", self.running_loss, self.global_step)
                        self.tb_writer.add_scalar("train/lr", lr, self.global_step)

                    if self.use_wandb:
                        import wandb
                        wandb.log(log_entry)

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint("checkpoint")
                    if self.running_loss < self.best_loss:
                        self.best_loss = self.running_loss
                        self.save_checkpoint("best")

        # Final save
        self.save_checkpoint("final")

        # Save training log
        log_path = self.output_dir / "train_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.train_log, f, indent=2)
        logger.info("Training complete. Final loss: %.4f", self.running_loss)

        if self.tb_writer:
            self.tb_writer.close()

        # Log peak VRAM
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info("Peak GPU VRAM: %.2f GB", peak_vram)


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="ZoomVid Trainer")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--module-type", type=str, default="dummy",
                        choices=list(MODULE_REGISTRY.keys()))
    parser.add_argument("--cache-dir", type=str, required=True, help="Feature cache directory")
    parser.add_argument("--train-manifest", type=str, required=True, help="Training manifest JSON")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config if provided
    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load backbone
    print("Loading frozen VLM backbone...")
    backbone = FrozenVLMBackbone(
        model_id=args.model_id,
        torch_dtype=args.dtype,
        device=device,
        load_pretrained=True,
    )

    in_feat = backbone.vision_feature_size
    out_feat = backbone.llm_hidden_size

    # Create module
    cls = MODULE_REGISTRY[args.module_type]
    module = cls(in_features=in_feat, out_features=out_feat)
    print(f"Module: {module.__class__.__name__} | Trainable params: {sum(p.numel() for p in module.parameters() if p.requires_grad):,}")

    # Load cache and manifest
    cache_reader = FeatureCacheReader(args.cache_dir)
    with open(args.train_manifest, "r", encoding="utf-8") as f:
        train_manifest = json.load(f)
    train_dataset = TrainingDataset(cache_reader, train_manifest)

    # Training config
    train_cfg = cfg.get("training", {})
    trainer = Trainer(
        module=module,
        backbone=backbone,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_steps=args.max_steps,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        budget_range=tuple(train_cfg.get("budget_range", [1024, 4096])),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        seed=train_cfg.get("seed", 42),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()

