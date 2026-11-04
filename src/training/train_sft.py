import argparse
import json
import logging
import os
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.training.sft_dataset import validate_sft_records

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for SQL Synthesis"
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
        default="checkpoints/sft_adapter",
        help="Adapter output dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["sft"]

    logger.info("Loading training dataset from %s...", config["dataset_path"])
    with open(config["dataset_path"], "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    valid_data, _ = validate_sft_records(raw_data)
    logger.info("Loaded %d valid training records after validation.", len(valid_data))

    tokenizer: Any = AutoTokenizer.from_pretrained(
        config["base_model"], trust_remote_code=True
    )
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for {config['base_model']}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_texts: list[str] = []
    for item in valid_data:
        messages = [
            {"role": "user", "content": item.get("input", "")},
            {"role": "assistant", "content": item.get("output", "")},
        ]
        formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

    dataset = Dataset.from_dict({"text": formatted_texts})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model: %s...", config["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.get("lora_r", 32),
        lora_alpha=config.get("lora_alpha", 64),
        lora_dropout=config.get("lora_dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=config.get("num_train_epochs", 2),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_ratio", 0.05),
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_length=config.get("max_seq_length", 8192),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info("Saving trained adapter to %s...", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if trainer.model is not None:
        trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("SFT training pipeline finished successfully.")


if __name__ == "__main__":
    main()
