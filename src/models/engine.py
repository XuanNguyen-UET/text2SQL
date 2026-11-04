# src/models/engine.py
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Applies tokenizer chat template with compatibility fallbacks."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


class HuggingFaceEmbedder:
    """Native PyTorch embedding engine for dense schema representation."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs[0]
                attention_mask = (
                    encoded["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
                normalized = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

        return np.vstack(all_embeddings)


class VLLMRunner:
    """High-throughput inference engine powered by vLLM."""

    def __init__(
        self,
        model_id: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.5,
        tensor_parallel_size: int = 1,
    ) -> None:
        from vllm import LLM

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        n: int = 1,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        stop_token_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id > 0:
            stop_token_ids.append(im_end_id)

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or [],
            stop_token_ids=stop_token_ids,
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        results: list[list[str]] = []
        for output in outputs:
            results.append([cand.text for cand in output.outputs])
        return results


class HFQuantizedRunner:
    """Low-memory 4-bit NF4 quantized inference engine using BitsAndBytes."""

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0.0

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if stop_tokens:
            for stop in stop_tokens:
                if stop in output_text:
                    output_text = output_text.split(stop)[0]

        return output_text.strip()
