import re

from src.models.engine import HFQuantizedRunner, VLLMRunner, apply_chat_template_safe
from src.models.prompts import LOGICAL_PLAN_SYSTEM_PROMPT, LOGICAL_PLAN_USER_TEMPLATE


def clean_reasoning_tags(text: str) -> str:
    """Removes thinking blocks and extracts final logical plan steps."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<answer>|</answer>", "", cleaned)
    return cleaned.strip()


class LogicalPlanner:
    """Uses a 4B SLM to generate abstract, schema-agnostic reasoning plans."""

    def __init__(self, runner: VLLMRunner | HFQuantizedRunner) -> None:
        self.runner = runner

    def build_prompt(self, question: str, evidence: str = "") -> str:
        user_content = LOGICAL_PLAN_USER_TEMPLATE.format(
            question=question,
            evidence=evidence or "None",
        )
        messages = [
            {"role": "system", "content": LOGICAL_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return apply_chat_template_safe(
            self.runner.tokenizer, messages, add_generation_prompt=True
        )

    def plan_single(
        self, question: str, evidence: str = "", max_tokens: int = 512
    ) -> str:
        prompt = self.build_prompt(question, evidence)
        if isinstance(self.runner, VLLMRunner):
            outputs = self.runner.generate(
                [prompt], max_tokens=max_tokens, temperature=0.0
            )
            raw_text = outputs[0][0]
        else:
            raw_text = self.runner.generate(
                prompt, max_new_tokens=max_tokens, temperature=0.0
            )

        return clean_reasoning_tags(raw_text)

    def plan_batch(
        self,
        questions: list[str],
        evidences: list[str] | None = None,
        max_tokens: int = 512,
    ) -> list[str]:
        if evidences is None:
            evidences = ["" for _ in questions]

        prompts = [self.build_prompt(q, ev) for q, ev in zip(questions, evidences)]

        if isinstance(self.runner, VLLMRunner):
            outputs = self.runner.generate(
                prompts, max_tokens=max_tokens, temperature=0.0
            )
            return [clean_reasoning_tags(cands[0]) for cands in outputs]

        return [
            clean_reasoning_tags(
                self.runner.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
            )
            for prompt in prompts
        ]
