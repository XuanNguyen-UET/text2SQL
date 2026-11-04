from dataclasses import dataclass, field

from src.sandbox.executor import ExecutionResult


@dataclass(slots=True)
class CandidateSQL:
    sql: str
    raw_response: str = ""
    execution_result: ExecutionResult | None = None
    is_valid_format: bool = True


@dataclass(slots=True)
class AgentStepTrace:
    iteration: int
    action_taken: str
    reasoning: str
    target_candidate_index: int | None = None
    candidates: list[CandidateSQL] = field(default_factory=list)
    consensus_reached: bool = False


@dataclass(slots=True)
class AgentState:
    db_id: str
    question: str
    evidence: str = ""
    logical_plan: str = ""
    pruned_schema: dict[str, list[str]] = field(default_factory=dict)
    mini_ddl: str = ""
    current_iteration: int = 0
    max_iterations: int = 2
    traces: list[AgentStepTrace] = field(default_factory=list)
    final_sql: str = ""
    status: str = "INITIALIZED"
