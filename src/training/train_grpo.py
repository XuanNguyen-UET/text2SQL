import argparse
import json
import logging
import os
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from src.training.rewards import (
    execution_reward_func,
    format_reward_func,
    match_reward_func,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_grpo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO Reinforcement Learning for SQL Synthesis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to train YAML config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/grpo_final",
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
        grpo_cfg = full_cfg["grpo"]
        sft_cfg = full_cfg["sft"]

    logger.info("Loading GRPO dataset from %s...", grpo_cfg["dataset_path"])
    with open(grpo_cfg["dataset_path"], "r", encoding="utf-8") as f:
        raw_grpo = json.load(f)

    prompts: list[str] = [item["prompt"] for item in raw_grpo]
    gold_sqls: list[str] = [item.get("gold_sql", "") for item in raw_grpo]
    db_paths: list[str] = [item.get("db_path", "") for item in raw_grpo]

    dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "gold_sql": gold_sqls,
            "db_path": db_paths,
        }
    )

    base_model_name = sft_cfg["base_model"]
    tokenizer: Any = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for {base_model_name}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s...", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    sft_checkpoint = grpo_cfg.get("base_checkpoint", "")
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        logger.info("Attaching SFT adapter from %s...", sft_checkpoint)
        model = PeftModel.from_pretrained(base_model, sft_checkpoint, is_trainable=True)
    else:
        logger.warning(
            "SFT checkpoint not found at %s. Training on base model directly.",
            sft_checkpoint,
        )
        model = base_model

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=float(grpo_cfg.get("learning_rate", 2e-5)),
        max_steps=grpo_cfg.get("max_steps", 500),
        num_generations=grpo_cfg.get("num_generations", 4),
        per_device_train_batch_size=grpo_cfg.get("generation_batch_size", 2),
        gradient_accumulation_steps=2,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    reward_funcs: list[Any] = [
        format_reward_func,
        execution_reward_func,
        match_reward_func,
    ]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting GRPO reinforcement training...")
    trainer.train()

    logger.info("Saving GRPO model checkpoint to %s...", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if trainer.model is not None:
        trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("GRPO training finished successfully.")


if __name__ == "__main__":
    main()
