"""Ten-exchange demonstration using the real extraction, index, retrieval, and answer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from answer_generator import GroundedAnswer
from config import AppConfig, PROJECT_ROOT, get_config
from conversation_memory import ConversationMemory
from embedding_service import EmbeddingService, create_embedding_service
from runtime import AssistantRuntime

DEMO_DIR = PROJECT_ROOT / "demo_documents"
SAMPLE_FILES = [
    DEMO_DIR / "employee_onboarding_leave_policy.txt",
    DEMO_DIR / "product_support_troubleshooting_guide.txt",
]


@dataclass
class DemoExchange:
    number: int
    label: str
    question: str
    expected_documents: list[str]
    must_be_unanswerable: bool = False


DEMO_EXCHANGES: list[DemoExchange] = [
    DemoExchange(
        1,
        "Direct document question",
        "How many annual leave days do full-time employees receive after probation?",
        ["employee_onboarding_leave_policy.txt"],
    ),
    DemoExchange(
        2,
        "Follow-up depending on the previous answer",
        "Can unused leave be carried over to the next year?",
        ["employee_onboarding_leave_policy.txt"],
    ),
    DemoExchange(
        3,
        "Clarification request",
        "What does that mean for someone who still has 8 unused annual leave days in December?",
        ["employee_onboarding_leave_policy.txt"],
    ),
    DemoExchange(
        4,
        "Switch to a topic in another document",
        "What does error HD-441 mean in HelioDesk 4.2?",
        ["product_support_troubleshooting_guide.txt"],
    ),
    DemoExchange(
        5,
        "Return to the earlier topic",
        "Going back to the leave policy, how far in advance should planned annual leave be requested?",
        ["employee_onboarding_leave_policy.txt"],
    ),
    DemoExchange(
        6,
        "Question not covered by the documents",
        "What is the company's Friday cafeteria menu and the CEO salary band?",
        [],
        must_be_unanswerable=True,
    ),
    DemoExchange(
        7,
        "Question whose answer was already provided",
        "How many annual leave days do full-time employees receive after probation?",
        ["employee_onboarding_leave_policy.txt"],
    ),
    DemoExchange(
        8,
        "Summary request covering earlier discussion",
        "Please summarize what we have discussed so far.",
        [],
    ),
    DemoExchange(
        9,
        "Comparison between two document sections",
        "Compare the first-response time for P1 tickets with the resignation notice period for individual contributors.",
        ["product_support_troubleshooting_guide.txt", "employee_onboarding_leave_policy.txt"],
    ),
    DemoExchange(
        10,
        "Final question testing context retention",
        "If a new hire is still in probation, can they work from home two days a week, and does HD-902 change that rule?",
        ["employee_onboarding_leave_policy.txt", "product_support_troubleshooting_guide.txt"],
    ),
]


@dataclass
class ExchangeResult:
    spec: DemoExchange
    answer: GroundedAnswer | None
    question: str
    executed: bool
    citations_valid: bool
    notes: str = ""


@dataclass
class DemoRun:
    executed: bool
    generated_answers: bool
    indexed_files: list[str]
    exchanges: list[ExchangeResult] = field(default_factory=list)
    error: str | None = None
    status_message: str = ""
    memory: ConversationMemory | None = None

    @property
    def total(self) -> int:
        return len(DEMO_EXCHANGES)

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self.exchanges if item.executed)


def sample_document_texts() -> dict[str, str]:
    texts = {}
    for path in SAMPLE_FILES:
        texts[path.name] = path.read_text(encoding="utf-8")
    return texts


def load_demo_runtime(
    embeddings: EmbeddingService | None = None,
    config: AppConfig | None = None,
    use_deterministic_embeddings: bool = False,
) -> AssistantRuntime:
    config = config or get_config()
    if embeddings is None:
        embeddings = create_embedding_service(config.embedding_model, use_deterministic=use_deterministic_embeddings)
        if use_deterministic_embeddings:
            # Keep retrieval usable in tests without downloading a transformer.
            pass
        elif hasattr(embeddings, "dim"):
            pass
    runtime = AssistantRuntime(embeddings=embeddings, config=config)
    for path in SAMPLE_FILES:
        result = runtime.process_upload(path.read_bytes(), path.name, replace=True)
        if result.status != "indexed":
            raise RuntimeError(f"Failed to index sample document {path.name}: {result.message}")
    return runtime


def run_demonstration(
    embeddings: EmbeddingService | None = None,
    config: AppConfig | None = None,
    use_deterministic_embeddings: bool = False,
    runtime: AssistantRuntime | None = None,
    memory: ConversationMemory | None = None,
) -> DemoRun:
    """Execute all ten exchanges through the real application pipeline."""
    try:
        runtime = runtime or load_demo_runtime(
            embeddings=embeddings,
            config=config,
            use_deterministic_embeddings=use_deterministic_embeddings,
        )
    except Exception as exc:
        return DemoRun(
            executed=False,
            generated_answers=False,
            indexed_files=[],
            error=str(exc),
            status_message="Demonstration could not start because sample documents failed to index.",
        )

    memory = memory or ConversationMemory()
    results: list[ExchangeResult] = []
    generated = False
    for spec in DEMO_EXCHANGES:
        answer = runtime.ask(spec.question, memory)
        generated = generated or bool(answer.llm_used)
        citations_ok = True
        for citation in answer.citations:
            if not runtime.store.contains_chunk(citation.document_id, citation.chunk_id):
                citations_ok = False
            if spec.must_be_unanswerable:
                citations_ok = False
        if spec.must_be_unanswerable:
            citations_ok = len(answer.citations) == 0
        notes = ""
        if spec.must_be_unanswerable and answer.mode != "insufficient":
            notes = "Expected an insufficient-evidence outcome for this uncovered question."
        results.append(
            ExchangeResult(
                spec=spec,
                answer=answer,
                question=spec.question,
                executed=True,
                citations_valid=citations_ok,
                notes=notes,
            )
        )

    if generated:
        status = (
            f"Demonstration executed through the live pipeline with generated answers "
            f"({len(results)}/{len(DEMO_EXCHANGES)} exchanges)."
        )
    else:
        status = (
            "Demonstration retrieved evidence through the live pipeline. "
            "Full generated-answer execution requires a configured LLM API key. "
            "This is not a hardcoded transcript, but answers are evidence-only until the LLM is configured."
        )
    return DemoRun(
        executed=True,
        generated_answers=generated,
        indexed_files=[path.name for path in SAMPLE_FILES],
        exchanges=results,
        status_message=status,
        memory=memory,
    )
