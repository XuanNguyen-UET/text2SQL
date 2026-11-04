import argparse
import json
import logging
import os
import sys

from src.agent.orchestrator import SQLOrchestrator
from src.models.engine import HFQuantizedRunner, HuggingFaceEmbedder, VLLMRunner
from src.schema_linking.profiler import DatabaseProfiler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end Text-to-SQL Pipeline Inference"
    )
    parser.add_argument(
        "--db_id", type=str, default="financial", help="Database identifier"
    )
    parser.add_argument(
        "--question", type=str, default="", help="Natural language question"
    )
    parser.add_argument(
        "--evidence", type=str, default="", help="External schema evidence"
    )
    parser.add_argument(
        "--profiles_path", type=str, default="db_info/dev_profiles_explicit.json"
    )
    parser.add_argument("--db_dir", type=str, default="data/sample_databases")
    parser.add_argument(
        "--backend", type=str, choices=["vllm", "transformers_4bit"], default="vllm"
    )
    parser.add_argument("--slm_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--sql_model", type=str, default="NguyenCX/Qwen2.5_coder_7B_instruct_GRPO_BIRD"
    )
    parser.add_argument(
        "--output_json", type=str, default="outputs/predict_result.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.question:
        logger.error("Question must not be empty. Please provide --question.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    logger.info("Loading database profiles from %s", args.profiles_path)
    profiler = DatabaseProfiler(profiles_path=args.profiles_path)

    logger.info("Initializing embedding engine (BGE-small)...")
    embedder = HuggingFaceEmbedder()

    logger.info("Initializing models via backend: %s", args.backend)
    if args.backend == "vllm":
        slm_runner = VLLMRunner(model_id=args.slm_model, gpu_memory_utilization=0.28)
        sql_runner = VLLMRunner(model_id=args.sql_model, gpu_memory_utilization=0.56)
    else:
        slm_runner = HFQuantizedRunner(model_id=args.slm_model)
        sql_runner = HFQuantizedRunner(model_id=args.sql_model)

    orchestrator = SQLOrchestrator(
        profiler=profiler,
        sql_generator=sql_runner,
        slm_runner=slm_runner,
        embedder=embedder,
        db_directory=args.db_dir,
    )

    logger.info("Running pipeline for DB '%s'...", args.db_id)
    state = orchestrator.run(
        db_id=args.db_id, question=args.question, evidence=args.evidence
    )

    logger.info("Final State: %s", state.status)
    logger.info("Final SQL: %s", state.final_sql)

    output_payload = {
        "db_id": state.db_id,
        "question": state.question,
        "logical_plan": state.logical_plan,
        "mini_ddl": state.mini_ddl,
        "final_sql": state.final_sql,
        "status": state.status,
        "num_traces": len(state.traces),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    logger.info("Output successfully written to %s", args.output_json)


if __name__ == "__main__":
    main()
