import os

import gradio as gr
import pandas as pd

from src.agent.orchestrator import SQLOrchestrator
from src.models.engine import HFQuantizedRunner, HuggingFaceEmbedder, VLLMRunner
from src.sandbox.executor import execute_sql
from src.schema_linking.profiler import DatabaseProfiler

PROFILES_PATH = os.environ.get("PROFILES_PATH", "db_info/dev_profiles_explicit.json")
DB_DIR = os.environ.get("DB_DIR", "data/sample_databases")
BACKEND = os.environ.get("BACKEND", "vllm")
SLM_MODEL = os.environ.get("SLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
SQL_MODEL = os.environ.get("SQL_MODEL", "NguyenCX/Qwen2.5_coder_7B_instruct_GRPO_BIRD")

print("[INFO] Initializing Profiler and Embedder...")
profiler = DatabaseProfiler(
    profiles_path=PROFILES_PATH if os.path.exists(PROFILES_PATH) else None
)
embedder = HuggingFaceEmbedder()

print(f"[INFO] Initializing Runners via {BACKEND}...")
if BACKEND == "vllm":
    slm_runner = VLLMRunner(model_id=SLM_MODEL, gpu_memory_utilization=0.28)
    sql_runner = VLLMRunner(model_id=SQL_MODEL, gpu_memory_utilization=0.56)
else:
    slm_runner = HFQuantizedRunner(model_id=SLM_MODEL)
    sql_runner = HFQuantizedRunner(model_id=SQL_MODEL)

orchestrator = SQLOrchestrator(
    profiler=profiler,
    sql_generator=sql_runner,
    slm_runner=slm_runner,
    embedder=embedder,
    db_directory=DB_DIR,
)


def run_agent_pipeline(
    db_id: str,
    question: str,
    evidence: str,
    gold_sql: str = "",
) -> tuple[str, str, str, pd.DataFrame, pd.DataFrame, str]:
    if not question.strip():
        return (
            "",
            "",
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            "Please provide a valid question.",
        )

    state = orchestrator.run(
        db_id=db_id.strip(), question=question.strip(), evidence=evidence.strip()
    )

    cand_records: list[dict[str, str]] = []
    if state.traces:
        last_trace = state.traces[-1]
        for idx, cand in enumerate(last_trace.candidates):
            status = (
                cand.execution_result.status if cand.execution_result else "UNKNOWN"
            )
            cand_records.append(
                {
                    "Candidate": f"Candidate #{idx + 1}",
                    "SQL": cand.sql,
                    "Status": status,
                }
            )
    cand_df = pd.DataFrame(cand_records)

    db_path = orchestrator._resolve_db_path(db_id)
    pred_res = execute_sql(db_path, state.final_sql)

    res_df = pd.DataFrame()
    if pred_res.status == "SUCCESS" and pred_res.result:
        res_df = pd.DataFrame(pred_res.result).head(20)

    comparison_info = (
        f"Execution Status: {pred_res.status} ({pred_res.execution_time:.3f}s)"
    )
    if gold_sql.strip():
        gold_res = execute_sql(db_path, gold_sql.strip())
        is_match = (pred_res.status == "SUCCESS") and (
            pred_res.result == gold_res.result
        )
        comparison_info += f" | Gold SQL Match: {'MATCHED' if is_match else 'MISMATCH'}"

    return (
        state.logical_plan,
        state.mini_ddl,
        state.final_sql,
        cand_df,
        res_df,
        comparison_info,
    )


with gr.Blocks(title="Stateful Text-to-SQL Multi-Model Agent") as demo:
    gr.Markdown("# Stateful Execution-Guided Text-to-SQL Agent")
    gr.Markdown(
        "Heterogeneous SLM (4B) & Code-LLM (7B) with Sandbox Execution Guidance"
    )

    with gr.Row():
        with gr.Column(scale=1):
            db_id_input = gr.Textbox(label="Database ID", value="financial")
            question_input = gr.Textbox(
                label="Question",
                value="Which account has the highest frequency of transactions?",
                lines=3,
            )
            evidence_input = gr.Textbox(
                label="Evidence",
                value="frequency can be obtained from trans table count",
                lines=2,
            )
            gold_sql_input = gr.Textbox(label="Gold SQL (Optional Evaluation)", lines=2)
            submit_btn = gr.Button("Run Agent", variant="primary")

        with gr.Column(scale=2):
            with gr.Tab("Final SQL & Results"):
                final_sql_out = gr.Code(label="Final Selected SQL", language="sql")
                feedback_out = gr.Textbox(label="Execution Feedback", interactive=False)
                results_table = gr.Dataframe(
                    label="Sandbox Execution Results (First 20 rows)"
                )

            with gr.Tab("Agent Reasoning Traces"):
                logical_plan_out = gr.Markdown(label="Abstract Logical Plan")
                mini_ddl_out = gr.Code(label="Compressed Mini-DDL", language="sql")
                candidates_table = gr.Dataframe(label="Candidate Pool & Consensus")

    submit_btn.click(
        fn=run_agent_pipeline,
        inputs=[db_id_input, question_input, evidence_input, gold_sql_input],
        outputs=[
            logical_plan_out,
            mini_ddl_out,
            final_sql_out,
            candidates_table,
            results_table,
            feedback_out,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
