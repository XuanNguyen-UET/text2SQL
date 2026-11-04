import json
import logging
import os
import re
from typing import Any

from src.agent.state import AgentState, AgentStepTrace, CandidateSQL
from src.models.engine import (
    HFQuantizedRunner,
    HuggingFaceEmbedder,
    VLLMRunner,
    apply_chat_template_safe,
)
from src.models.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_USER_TEMPLATE,
    SQL_GENERATION_SYSTEM_PROMPT,
    SQL_GENERATION_USER_TEMPLATE,
    SQL_REPAIR_SYSTEM_PROMPT,
    SQL_REPAIR_USER_TEMPLATE,
)
from src.sandbox.executor import ExecutionResult, execute_sql
from src.sandbox.summarizer import summarize_execution_candidates
from src.schema_linking.ddl_rewriter import generate_mini_ddl
from src.schema_linking.planner import LogicalPlanner
from src.schema_linking.profiler import DatabaseProfile, DatabaseProfiler
from src.schema_linking.pruner import prune_schema

logger = logging.getLogger(__name__)


def extract_sql_from_response(raw_text: str) -> tuple[str, bool]:
    """Extracts SQL query strictly from <answer> tags, falling back to markdown block extraction."""
    answer_match = re.search(
        r"<answer>(.*?)</answer>", raw_text, re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        sql = answer_match.group(1).strip()
        sql = re.sub(r"^```sql|^```|```$", "", sql, flags=re.MULTILINE).strip()
        return sql, True

    code_match = re.search(r"```sql\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip(), False

    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return cleaned, False


class SQLOrchestrator:
    """Stateful execution-guided Text-to-SQL agent orchestrator."""

    def __init__(
        self,
        profiler: DatabaseProfiler,
        sql_generator: VLLMRunner | HFQuantizedRunner,
        slm_runner: VLLMRunner | HFQuantizedRunner,
        embedder: HuggingFaceEmbedder | None = None,
        db_directory: str = "data/sample_databases",
        schema_mode: str = "hybrid",
        num_candidates: int = 4,
        max_repair_iterations: int = 2,
        sandbox_timeout: float = 10.0,
    ) -> None:
        self.profiler = profiler
        self.sql_generator = sql_generator
        self.slm_runner = slm_runner
        self.embedder = embedder
        self.db_directory = db_directory
        self.schema_mode = schema_mode
        self.num_candidates = num_candidates
        self.max_repair_iterations = max_repair_iterations
        self.sandbox_timeout = sandbox_timeout
        self.planner = LogicalPlanner(runner=self.slm_runner)

    def _resolve_db_path(self, db_id: str) -> str:
        candidates = [
            os.path.join(self.db_directory, db_id, f"{db_id}.sqlite"),
            os.path.join(self.db_directory, f"{db_id}.sqlite"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _generate_candidate_sqls(
        self, mini_ddl: str, question: str, evidence: str
    ) -> list[CandidateSQL]:
        user_content = SQL_GENERATION_USER_TEMPLATE.format(
            mini_ddl=mini_ddl,
            question=question,
            evidence=evidence or "None",
        )
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = apply_chat_template_safe(
            self.sql_generator.tokenizer, messages, add_generation_prompt=True
        )

        raw_outputs: list[str] = []
        if isinstance(self.sql_generator, VLLMRunner):
            outputs = self.sql_generator.generate(
                [prompt],
                max_tokens=2048,
                temperature=0.7,
                n=self.num_candidates,
            )
            raw_outputs = outputs[0]
        else:
            for _ in range(self.num_candidates):
                out = self.sql_generator.generate(
                    prompt, max_new_tokens=2048, temperature=0.7
                )
                raw_outputs.append(out)

        candidates: list[CandidateSQL] = []
        for raw in raw_outputs:
            sql, is_valid = extract_sql_from_response(raw)
            candidates.append(
                CandidateSQL(sql=sql, raw_response=raw, is_valid_format=is_valid)
            )

        return candidates

    def _evaluate_candidates(
        self, db_path: str, candidates: list[CandidateSQL]
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for cand in candidates:
            res = execute_sql(db_path, cand.sql, timeout_seconds=self.sandbox_timeout)
            cand.execution_result = res
            results.append(res)
        return results

    def _consult_slm_supervisor(
        self,
        question: str,
        execution_summary: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        user_content = ORCHESTRATOR_USER_TEMPLATE.format(
            question=question,
            execution_summary=json.dumps(execution_summary, indent=2),
            iteration=iteration,
            max_iterations=self.max_repair_iterations,
        )
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = apply_chat_template_safe(
            self.slm_runner.tokenizer, messages, add_generation_prompt=True
        )

        if isinstance(self.slm_runner, VLLMRunner):
            raw_out = self.slm_runner.generate(
                [prompt], max_tokens=512, temperature=0.0
            )[0][0]
        else:
            raw_out = self.slm_runner.generate(
                prompt, max_new_tokens=512, temperature=0.0
            )

        match = re.search(r"\{.*?\}", raw_out, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
            except json.JSONDecodeError as exc:
                logger.warning(
                    "SLM supervisor emitted invalid JSON: %s. Raw output: %s",
                    exc,
                    raw_out,
                )

        fallback_action = (
            "VOTE_SELECT"
            if execution_summary.get("consensus_reached")
            else "STOP_WITH_FALLBACK"
        )
        logger.info(
            "Supervisor fallback activated. Assigned action: %s", fallback_action
        )
        return {
            "action": fallback_action,
            "target_index": None,
            "reason": "Supervisor JSON parsing failed; applied heuristic consensus rule.",
        }

    def _repair_sql(
        self,
        mini_ddl: str,
        question: str,
        evidence: str,
        failed_sql: str,
        error_trace: str,
    ) -> CandidateSQL:
        user_content = SQL_REPAIR_USER_TEMPLATE.format(
            mini_ddl=mini_ddl,
            question=question,
            evidence=evidence or "None",
            failed_sql=failed_sql,
            error_trace=error_trace,
        )
        messages = [
            {"role": "system", "content": SQL_REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = apply_chat_template_safe(
            self.sql_generator.tokenizer, messages, add_generation_prompt=True
        )

        if isinstance(self.sql_generator, VLLMRunner):
            raw_out = self.sql_generator.generate(
                [prompt], max_tokens=2048, temperature=0.2
            )[0][0]
        else:
            raw_out = self.sql_generator.generate(
                prompt, max_new_tokens=2048, temperature=0.2
            )

        sql, is_valid = extract_sql_from_response(raw_out)
        return CandidateSQL(sql=sql, raw_response=raw_out, is_valid_format=is_valid)

    def run(self, db_id: str, question: str, evidence: str = "") -> AgentState:
        state = AgentState(
            db_id=db_id,
            question=question,
            evidence=evidence,
            max_iterations=self.max_repair_iterations,
        )
        state.status = "RUNNING"

        profile = self.profiler.get_profile(db_id)
        if profile is None:
            profile = DatabaseProfile(db_id=db_id, tables=[], columns={})

        db_path = self._resolve_db_path(db_id)

        # Stage 1: Abstract Reasoning Plan
        state.logical_plan = self.planner.plan_single(question, evidence)

        # Stage 2: Schema Pruning & Mini-DDL
        state.pruned_schema = prune_schema(
            mode=self.schema_mode,
            profile=profile,
            question=question,
            evidence=evidence,
            embedder=self.embedder,
            db_path=db_path,
        )
        state.mini_ddl = generate_mini_ddl(
            runner=self.slm_runner,
            profile=profile,
            pruned_schema=state.pruned_schema,
            question=question,
            evidence=evidence,
            logical_plan=state.logical_plan,
        )

        # Stage 3: Initial Candidate Generation
        candidates = self._generate_candidate_sqls(state.mini_ddl, question, evidence)
        exec_results = self._evaluate_candidates(db_path, candidates)
        summary = summarize_execution_candidates(
            [c.sql for c in candidates], exec_results
        )

        for iteration in range(1, self.max_repair_iterations + 1):
            state.current_iteration = iteration

            # Fast path: Immediate consensus reached
            if summary["consensus_reached"]:
                trace = AgentStepTrace(
                    iteration=iteration,
                    action_taken="VOTE_SELECT",
                    reasoning=f"High-confidence consensus reached across {summary['dominant_cluster_size']} candidates.",
                    candidates=candidates,
                    consensus_reached=True,
                )
                state.traces.append(trace)
                state.final_sql = summary["recommended_sql"]
                state.status = "SUCCESS"
                return state

            decision = self._consult_slm_supervisor(question, summary, iteration)
            action = decision.get("action", "STOP_WITH_FALLBACK")
            reason = decision.get("reason", "No explanation provided.")
            target_idx = decision.get("target_index")

            trace = AgentStepTrace(
                iteration=iteration,
                action_taken=action,
                reasoning=reason,
                target_candidate_index=target_idx,
                candidates=list(candidates),
                consensus_reached=False,
            )
            state.traces.append(trace)

            if action == "VOTE_SELECT" and summary["recommended_sql"]:
                state.final_sql = summary["recommended_sql"]
                state.status = "SUCCESS"
                return state

            if (
                action == "REPAIR_SQL"
                and summary["errors"]
                and iteration < self.max_repair_iterations
            ):
                idx_to_repair = (
                    target_idx
                    if (target_idx is not None and target_idx in summary["errors"])
                    else next(iter(summary["errors"].keys()))
                )
                failed_cand = candidates[idx_to_repair]
                error_trace = summary["errors"][idx_to_repair]

                repaired_cand = self._repair_sql(
                    state.mini_ddl, question, evidence, failed_cand.sql, error_trace
                )
                repaired_res = execute_sql(
                    db_path, repaired_cand.sql, timeout_seconds=self.sandbox_timeout
                )
                repaired_cand.execution_result = repaired_res

                candidates[idx_to_repair] = repaired_cand
                exec_results[idx_to_repair] = repaired_res
                summary = summarize_execution_candidates(
                    [c.sql for c in candidates], exec_results
                )
                continue

            if action == "REGENERATE_SQL" and iteration < self.max_repair_iterations:
                candidates = self._generate_candidate_sqls(
                    state.mini_ddl, question, evidence
                )
                exec_results = self._evaluate_candidates(db_path, candidates)
                summary = summarize_execution_candidates(
                    [c.sql for c in candidates], exec_results
                )
                continue

            break

        # Fallback termination
        state.final_sql = summary.get("recommended_sql") or (
            candidates[0].sql if candidates else "SELECT 1;"
        )
        state.status = "FALLBACK" if not summary.get("recommended_sql") else "SUCCESS"
        return state
