"""REUSABLE DECISION PATTERN: evidence policy  (#policy)

What it does
    Evaluates whether collected evidence is relevant, sufficient, trustworthy, and free of
    irreconcilable contradictions before generating an answer to a technical research query.
    Enforces a strict evidence policy that maps calibrated probabilities into three programmatic actions:
    - `answer`: sufficient, relevant, and non-contradictory evidence supports a factual response citing source IDs.
    - `retrieve`: evidence is missing, partial, or insufficient, and an additional retrieval attempt is warranted.
    - `abstain`: evidence is completely absent, out-of-scope, or irreconcilably contradictory across sources.

Typed questions and options
    `evidence_sufficient` (noul): does the collected evidence contain relevant and sufficient factual information?
        {"true": "evidence contains the facts needed to answer", "false": "key facts are missing, incomplete, or irrelevant"}
    `contradictory` (noul): do the evidence sources present conflicting or mutually contradictory claims?
        {"true": "sources assert opposing findings or disputed data", "false": "sources agree or do not conflict"}
    `evidence_action` (choice): under the evidence policy, what action should be taken next?
        {"answer": "sufficient and consistent evidence supports answering",
         "retrieve": "evidence is insufficient but another retrieval attempt is justified",
         "abstain": "evidence is permanently absent or irreconcilably contradictory"}

Wiring
    EvidencePolicyCheck sits in the agent's research loop after evidence retrieval and before synthesis:
        policy = EvidencePolicyCheck(decider)
        verdict = policy.check(query, evidence, retrieval_count, max_retrievals)
        if verdict.action == "retrieve":
            # initiate refined retrieval attempt
        elif verdict.action == "abstain":
            # abstain with transparent explanation of missing facts or contradiction
        else:
            # invoke Amazon Bedrock to synthesize answer citing [SRC-xxx]

How to reuse it
    Instantiate EvidencePolicyCheck with any Decider instance. Call check(query, evidence)
    with your user prompt and retrieved snippets. Customize thresholds in decide() to tune
    strictness on contradictory claims or evidence sufficiency for your domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decider_kit import Decider
from decider_kit.client import Decision

PATTERN_TAG = "#policy"

QUESTIONS = {
    "evidence_sufficient": Decider.noul(
        "Does the collected evidence contain relevant and sufficient factual information to answer the research query?",
        {
            "true": "the evidence contains the specific facts, measurements, or findings required to answer the query",
            "false": "the evidence lacks key facts, is incomplete, irrelevant, or fails to address the query",
        },
    ),
    "contradictory": Decider.noul(
        "Do the evidence sources present conflicting or mutually contradictory claims regarding the query?",
        {
            "true": "sources assert opposing findings, inconsistent data, or directly disputed claims",
            "false": "sources agree in their findings, corroborate each other, or do not conflict",
        },
    ),
    "evidence_action": Decider.choice(
        "Under the evidence policy, what action should be taken next?",
        {
            "answer": "sufficient and consistent evidence is present to generate a grounded, cited answer",
            "retrieve": "evidence is currently insufficient but another retrieval attempt is justified",
            "abstain": "evidence is permanently absent, irrelevant, or irreconcilably contradictory",
        },
    ),
}


@dataclass
class EvidenceVerdict:
    action: str  # "answer" | "retrieve" | "abstain"
    evidence_sufficient: float
    contradictory: float
    confidence: float
    reason: str
    answers: dict[str, Decision] = field(repr=False, default_factory=dict)
    decider_ms: float = 0.0

    @property
    def verdict(self) -> str:
        return self.action


def format_evidence_state(
    query: str,
    evidence: list[dict[str, Any]] | str,
    retrieval_count: int = 1,
    max_retrievals: int = 2,
) -> str:
    """Format research query, attempt count, and evidence snippets for Decider evaluation."""
    lines = [
        f"Research Query: {query.strip()}",
        f"Retrieval Attempt: {retrieval_count} of {max_retrievals}",
        "",
        "Retrieved Evidence:",
    ]
    if isinstance(evidence, str):
        lines.append(evidence.strip() if evidence.strip() else "(No evidence retrieved)")
    elif not evidence:
        lines.append("(No evidence retrieved)")
    else:
        for item in evidence:
            sid = item.get("source_id", "SRC-UNKNOWN")
            title = item.get("title", "Untitled Source")
            text = item.get("text", "")
            lines.append(f"- [{sid}] {title}: {text}")

    return "\n".join(lines)


class EvidencePolicyCheck:
    """Policy classifier evaluating evidence sufficiency and contradiction before answering."""

    def __init__(
        self,
        decider: Decider,
        sufficiency_threshold: float = 0.5,
        contradiction_threshold: float = 0.4,
    ):
        self.decider = decider
        self.sufficiency_threshold = sufficiency_threshold
        self.contradiction_threshold = contradiction_threshold
        self.log: list[dict[str, Any]] = []

    def format_state(
        self,
        query: str,
        evidence: list[dict[str, Any]] | str,
        retrieval_count: int = 1,
        max_retrievals: int = 2,
    ) -> str:
        return format_evidence_state(query, evidence, retrieval_count, max_retrievals)

    def check(
        self,
        query: str,
        evidence: list[dict[str, Any]] | str,
        retrieval_count: int = 1,
        max_retrievals: int = 2,
    ) -> EvidenceVerdict:
        state = self.format_state(query, evidence, retrieval_count, max_retrievals)
        answers = self.decider.ask(state, QUESTIONS)
        verdict = self.decide(answers, retrieval_count, max_retrievals)
        verdict.decider_ms = self.decider.last_latency_ms or 0.0

        rec = {
            "pattern": PATTERN_TAG,
            "stage": "evidence_policy",
            "action": verdict.action,
            "verdict": verdict.action,
            "evidence_sufficient": round(verdict.evidence_sufficient, 3),
            "contradictory": round(verdict.contradictory, 3),
            "confidence": round(verdict.confidence, 3),
            "reason": verdict.reason,
            "retrieval_count": retrieval_count,
            "max_retrievals": max_retrievals,
            "decider_ms": round(verdict.decider_ms),
        }
        self.log.append(rec)
        return verdict

    def decide(
        self,
        answers: dict[str, Decision],
        retrieval_count: int,
        max_retrievals: int,
    ) -> EvidenceVerdict:
        p_suff = answers["evidence_sufficient"].p_yes if answers["evidence_sufficient"].p_yes is not None else 0.0
        p_contra = answers["contradictory"].p_yes if answers["contradictory"].p_yes is not None else 0.0
        choice_act = answers["evidence_action"].label or "abstain"
        conf = answers["evidence_action"].confidence

        # Rule 1: Contradiction gating
        # If contradictory claims are detected between sources, abstain from picking an arbitrary winner.
        if p_contra >= self.contradiction_threshold or choice_act == "abstain" and p_contra > 0.3:
            if p_contra >= self.contradiction_threshold:
                return EvidenceVerdict(
                    action="abstain",
                    evidence_sufficient=p_suff,
                    contradictory=p_contra,
                    confidence=conf,
                    reason="Contradictory evidence detected across sources; policy mandates abstention.",
                    answers=answers,
                )

        # Rule 2: Sufficiency gating
        # If evidence is sufficient and non-contradictory, approve answer generation.
        if p_suff >= self.sufficiency_threshold and choice_act == "answer" and p_contra < self.contradiction_threshold:
            return EvidenceVerdict(
                action="answer",
                evidence_sufficient=p_suff,
                contradictory=p_contra,
                confidence=conf,
                reason="Sufficient and consistent evidence verified to answer query.",
                answers=answers,
            )

        # Rule 3: Retrieval loop boundary
        # If evidence is insufficient but additional attempts remain, trigger retrieval.
        if retrieval_count < max_retrievals and choice_act == "retrieve":
            return EvidenceVerdict(
                action="retrieve",
                evidence_sufficient=p_suff,
                contradictory=p_contra,
                confidence=conf,
                reason="Evidence is incomplete; initiating another retrieval attempt.",
                answers=answers,
            )

        # Rule 4: Exhausted attempts or permanently insufficient evidence
        return EvidenceVerdict(
            action="abstain",
            evidence_sufficient=p_suff,
            contradictory=p_contra,
            confidence=conf,
            reason="Insufficient evidence found in corpus after retrieval attempts.",
            answers=answers,
        )
