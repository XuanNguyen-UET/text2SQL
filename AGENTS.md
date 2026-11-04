# AGENTS.md — System Context & Engineering Guidelines

## 1. Project Overview
**Text2SQL-Agent** is a stateful, execution-guided single-agent Text-to-SQL framework designed for robust on-premise deployment[cite: 1]. Rather than relying on multiple autonomous conversational agents, the system operates as a **single unified agent** driving a deterministic state machine, routing tasks to **specialized heterogeneous models** tailored for each stage (evaluating on the BIRD Benchmark: **64.27% EX**, **68.67% VES**):
- **Dense Representation**: `BAAI/bge-small-en-v1.5` for semantic schema and table linking[cite: 5].
- **Logical Planning & Orchestration**: 4B SLM (`Qwen/Qwen3-4B-Instruct-2507` or `Qwen/Qwen3.5-4B`) for schema-agnostic query breakdown and transition routing[cite: 1, 6].
- **SQL Synthesis & Self-Repair**: Specialized 7B Code LLM (`Qwen/Qwen2.5-Coder-7B-Instruct` post-trained via SFT + GRPO) for constrained candidate generation and error-driven repair[cite: 1, 3].

### Core Architecture
1. **Context-Aware Schema Linking (`src/schema_linking/`)**:
   - **Abstract Logical Planner**: Leverages the 4B SLM to decompose natural language questions into schema-agnostic reasoning steps[cite: 1, 6].
   - **Dual-Mode Schema Pruner**:
     - *Hybrid Mode*: Integrates `BM25Okapi` ($0.6$) + `BAAI/bge-small-en-v1.5` cosine similarity ($0.4$) with dynamic thresholding ($\tau = 0.15$)[cite: 5, 7], 1-hop neighbor expansion, and **Steiner Tree approximation** (`networkx`) over foreign key graphs to guarantee join-path connectivity[cite: 5, 7], combined with SQLite `LIKE` probes for physical entity verification[cite: 5].
     - *Fast Mode*: In-memory `BM25Index` with rule-based heuristics (`+8.0` backtick match, `+4.0` column mention, `+2.0` table mention, `+10.0` sample match) and **BFS shortest-path** foreign key closure[cite: 1, 4].
   - **Mini-DDL Rewriter & Validator**: Compresses selected schemas into minimal `CREATE TABLE` blocks retaining sample value annotations, validated against physical database profiles with fallback to algorithmic DDL upon hallucination[cite: 1, 4].
2. **Two-Stage Model Post-Training (`src/training/`, `scripts/train/`)**:
   - **Stage 1 (Cold-Start SFT)**: QLoRA fine-tuning (`Unsloth` + `TRL SFTTrainer`, `r=32`, `lora_alpha=64`) on `Qwen/Qwen2.5-Coder-7B-Instruct` using response-only loss masking to enforce structured `<think>...</think><answer>...</answer>` reasoning paths[cite: 3].
   - **Stage 2 (GRPO RL)**: Group Relative Policy Optimization (`TRL GRPOTrainer`) guided by rule-based and execution rewards (format validity, read-only SQLite sandbox status, and canonical result-set equivalence against Gold SQL)[cite: 1]. Production weights: `NguyenCX/Qwen2.5_coder_7B_instruct_GRPO_BIRD`[cite: 1].
3. **Stateful Orchestration & Execution Sandbox (`src/sandbox/`, `src/agent/`)**:
   - **SQLite Execution Sandbox**: Isolated read-only connection (`mode=ro`), regex DML/DDL blocker, and C-level `set_progress_handler` execution deadlines[cite: 1].
   - **Consensus Summarizer**: Normalizes row ordering, clusters candidate queries into equivalence classes based on execution outputs, and isolates execution traces[cite: 1].
   - **Bounded SLM Orchestrator**: The 4B SLM inspects aggregated execution feedback to select deterministic transitions (`VOTE_SELECT`, `REPAIR_SQL`, `REGENERATE_SQL`, `STOP_WITH_BEST`, `STOP_WITH_FALLBACK`), strictly governed by runtime guardrails (`guard_action`)[cite: 1].

---

## 2. Repository Structure
```text
.
├── configs/                    # YAML configurations (model, agent, training)
├── db_info/                    # Pre-extracted schema metadata & sample values
├── data/                       # Sample SQLite databases & dev records for testing
├── docker/                     # CUDA runtime Dockerfile & docker-compose.yml
├── src/
│   ├── models/                 # vLLM, 4-bit HF BitsAndBytes, and native PyTorch BGE embedder
│   ├── schema_linking/         # Profiler, Logical Planner, Pruner (Hybrid & Fast), DDL Rewriter
│   ├── sandbox/                # Read-only SQLite executor and consensus result summarizer
│   ├── agent/                  # AgentState, SLM Orchestrator, Guardrails, Repair, Voter
│   └── training/               # SFT dataset formatting and GRPO reward functions
├── scripts/                    # CLI entrypoints (inference, batch eval, SRR metrics, train)
├── demo/                       # Gradio interactive web UI
└── tests/                      # Unit & integration test suite

```

---

## 3. Tooling & Commands (`uv` First)

This project strictly uses **`uv`** for dependency and environment management.

* **Sync environment**: `uv sync` (or `uv sync --extra train --extra dev`)
* **Run CLI pipeline**: `uv run python scripts/run_pipeline.py --qid 0`
* **Run SRR evaluation**: `uv run python scripts/eval_srr.py`
* **Launch Gradio demo**: `uv run python demo/app.py`
* **Run test suite**: `uv run pytest tests/ -v`
* **Lint & Format**: `uv run ruff check . --fix && uv run ruff format .`

---

## 4. Technical Invariants (CRITICAL — DO NOT VIOLATE)

1. **No `sentence-transformers` Dependency**:
* NEVER add `sentence-transformers` to `pyproject.toml` or import it in `src/`. It causes severe runtime conflicts with `vllm` and PyTorch C++ extensions.


* Always use the native PyTorch + `transformers` implementation (`HuggingFaceEmbedder` with `AutoModel`, attention-masked mean pooling, and `F.normalize`) for `BAAI/bge-small-en-v1.5`.




2. **Standardized Foreign Key Format**:
* Across all modules, `explicit_foreign_keys` in database profiles MUST follow the dictionary schema:
`[{"source": "source_table.source_col", "target": "target_table.target_col"}, ...]`


* Graph builders (`networkx` or BFS) must derive table-level edges by splitting on `.`.




3. **Strict Prompt Formatting**:
* Always use `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)` wrapped in a safe fallback helper for `enable_thinking=False`. Never manually concatenate `<|im_start|>` tags in inference code.




4. **Sandbox Safety**:
* All SQLite executions in `src/sandbox/executor.py` and `src/training/rewards.py` must enforce read-only validation (`SELECT`/`WITH` only), open connections in read-only URI mode (`file:{db_path}?mode=ro`), and attach a timeout mechanism (`conn.set_progress_handler`).




5. **Path Resolution**:
* Never hardcode environment-specific paths (e.g., `/content/drive/...`, `/kaggle/...`). Resolve all paths relative to the repository root using `pathlib.Path`, environment variables, or YAML config files.





---

## 5. Coding Conventions & Anti-AI-Slop Rules

When writing, refactoring, or generating code in this repository, strictly adhere to these rules:

1. **Zero Meta-Commentary**:
* Do NOT write comments explaining the history or merging of the code (e.g., *"Unified from Notebook 2"*, *"Changed to match demo_grpo.py"*, *"Added to fix compatibility"*).
* Write code as if the heterogeneous model routing and state architecture were designed from scratch as a single cohesive system.


2. **No Emojis or Chatty Logs**:
* Do NOT use emojis (`🚀`, `✅`, `🔥`, `⚠️`, `🟢`) in comments, docstrings, `print()` statements, or logger outputs.


* Use standard Python `logging` (`logger = logging.getLogger(__name__)`) with clean, structured messages (`logger.info("Loaded %d records", len(records))`).


3. **No Obvious Comments & No Placeholders**:
* Do not comment self-explanatory code (e.g., `# Load JSON file`). Reserve comments strictly for non-obvious algorithmic invariants (e.g., `# Return non-zero in SQLite progress callback to abort timed-out query`).


* Never leave `# TODO`, `pass`, or `...` placeholders in production modules. Every function must be fully implemented.


4. **Modern Python 3.10+ Typing**:
* Use built-in generic types (`list[str]`, `dict[str, Any]`, `tuple[int, int]`, `str | None`) instead of legacy `typing.List`, `typing.Dict`, `typing.Optional`.


* Use `@dataclass` for structured data transfer objects (`BM25Document`, `AgentState`, `ExecutionResult`).




5. **Import Ordering**:
* Group imports into three blocks separated by a blank line: (1) Standard library, (2) Third-party libraries, (3) Local `src.*` modules.



---

## 6. Git Workflow & Atomic Commit Policy

* Follow **Conventional Commits** (`feat:`, `fix:`, `chore:`, `build:`, `refactor:`, `test:`, `docs:`).
* Keep commits modular and scoped to a single subsystem (e.g., `feat(sandbox): ...`, `feat(schema): ...`, `feat(agent): ...`). Never dump unrelated subsystem changes into a single monolithic commit.
* Never commit model weights (`*.safetensors`, `*.bin`, `*.pt`), large SQLite databases outside `data/sample_databases/`, `.env` files, or `__pycache__/`.

```

```