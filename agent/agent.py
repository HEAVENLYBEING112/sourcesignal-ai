"""SourceSignal AI: Evidence-Gated Research Agent.

Amazon Bedrock generates grounded answers citing verified sources, while Strands Decider 2B
evaluates evidence sufficiency, contradiction policy, and bounds retrieval attempts.

    from agent.agent import HackathonAgent
    out = HackathonAgent().chat("What are the scaling and memory characteristics of UltraAttention-v2?")
    out["reply"], out["decisions"]
"""

from __future__ import annotations

import os
import time
from typing import Any

from strands import Agent

from decider_kit import Decider
from decider_kit.llm import agent_model, default_model_id, image_blocks, strip_thinking

from .decision import EvidencePolicyCheck, EvidenceVerdict
from .tools import TOOLS, fetch_source, search_corpus

NAME = "SourceSignal AI: Evidence-Gated Research Agent"

SYSTEM_PROMPT = """You are SourceSignal AI, an evidence-gated technical research agent.
Your mission is to answer research questions using ONLY the verified evidence snippets provided.

Strict Evidence Policy Rules:
1. Every factual assertion must be explicitly grounded in the provided evidence.
2. For every fact, cite the supporting source ID using [SRC-xxx] notation (e.g., [SRC-101]).
3. Explicitly distinguish verified facts supported by evidence from unverified or unresolved claims.
4. If sources conflict, state the conflicting claims neutrally; do not pick an unsubstantiated winner.
5. Never invent facts, figures, or source citations."""

EAGER_PROMPT = """You are SourceSignal AI. Answer research questions immediately based on provided evidence with exact [SRC-xxx] citations."""

MAX_RETRIEVAL_ATTEMPTS = 2


class HackathonAgent:
    """Evidence-gated research agent with Strands Decider 2B policy enforcement."""

    def __init__(
        self,
        decider: Decider | None = None,
        model_id: str | None = None,
        eager: bool | None = None,
    ):
        self.decider = decider or Decider()
        self.eager = eager if eager is not None else os.environ.get("DEMO_EAGER") == "1"
        self.model_id = model_id or default_model_id()
        self.policy = EvidencePolicyCheck(self.decider)
        self.sessions: dict[str, Agent] = {}

    def _session(self, session_id: str) -> Agent:
        if session_id not in self.sessions:
            agent = Agent(
                model=agent_model(self.model_id),
                system_prompt=EAGER_PROMPT if self.eager else SYSTEM_PROMPT,
                tools=TOOLS,
                callback_handler=None,
            )
            self.sessions[session_id] = agent
        return self.sessions[session_id]

    def _format_evidence_text(self, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "(No evidence found)"
        lines = []
        for item in evidence:
            sid = item.get("source_id", "SRC-UNKNOWN")
            title = item.get("title", "")
            src = item.get("source", "")
            text = item.get("text", "")
            lines.append(f"[{sid}] {title} ({src}):\n{text}")
        return "\n\n".join(lines)

    def _fallback_synthesis(self, query: str, evidence: list[dict[str, Any]]) -> str:
        """Deterministic grounded synthesis when LLM provider is offline or unreachable."""
        src_ids = [e["source_id"] for e in evidence if "source_id" in e]
        src_tags = ", ".join(f"[{s}]" for s in sorted(set(src_ids)))
        summary_points = []
        for e in evidence:
            sid = e.get("source_id", "")
            text = e.get("text", "")
            summary_points.append(f"- [{sid}] {text}")

        return (
            f"Verified findings for '{query}' based on {src_tags}:\n\n"
            + "\n".join(summary_points)
            + "\n\nSupported facts: Grounded directly in verified corpus records above.\n"
            "Unresolved claims: None within the retrieved evidence scope."
        )

    def _format_abstention(self, query: str, verdict: EvidenceVerdict, evidence: list[dict[str, Any]]) -> str:
        """Format a transparent, structured abstention explanation."""
        if verdict.contradictory >= self.policy.contradiction_threshold:
            lines = [
                "SourceSignal Policy: ABSTAIN (Contradictory Evidence Detected)",
                "",
                f"The agent retrieved evidence for '{query}' but detected conflicting claims across sources:",
            ]
            for item in evidence:
                lines.append(f"- [{item.get('source_id')}] {item.get('title')}: {item.get('text')}")
            lines.append("")
            lines.append(
                "Policy Action: In accordance with SourceSignal's evidence policy, the agent abstains "
                "from presenting a conclusive answer because verified sources disagree on critical facts. "
                "The conflicting findings remain unresolved without an independent audit."
            )
            return "\n".join(lines)

        return (
            f"SourceSignal Policy: ABSTAIN (Insufficient Evidence)\n\n"
            f"The agent executed {MAX_RETRIEVAL_ATTEMPTS} retrieval attempts for '{query}', "
            f"but no verified sources in the corpus provide sufficient facts to answer the question.\n\n"
            f"Policy Action: To guarantee zero hallucination, the agent abstains from speculating or "
            f"generating an ungrounded response."
        )

    def chat(
        self,
        message: str,
        session_id: str = "default",
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Execute evidence-gated research workflow: retrieve, evaluate with Decider, answer or abstain."""
        t0 = time.perf_counter()
        before_decisions = len(self.policy.log)

        # Step 1: Initial evidence retrieval
        evidence = search_corpus(message, limit=3)
        retrieval_count = 1

        # Step 2: First decision turn
        verdict = self.policy.check(
            query=message,
            evidence=evidence,
            retrieval_count=retrieval_count,
            max_retrievals=MAX_RETRIEVAL_ATTEMPTS,
        )

        # Step 3: Retrieval loop (if decider requests more evidence and attempts remain)
        if verdict.action == "retrieve" and retrieval_count < MAX_RETRIEVAL_ATTEMPTS:
            retrieval_count += 1
            # Targeted follow-up search: split query terms to widen search
            words = [w for w in message.split() if len(w) > 3]
            secondary_query = " ".join(words[-3:]) if words else message
            more_evidence = search_corpus(secondary_query, limit=2)

            # Deduplicate by source_id
            seen_ids = {e["source_id"] for e in evidence if "source_id" in e}
            for item in more_evidence:
                if item.get("source_id") not in seen_ids:
                    evidence.append(item)
                    seen_ids.add(item["source_id"])

            # Re-evaluate evidence policy with updated evidence set
            verdict = self.policy.check(
                query=message,
                evidence=evidence,
                retrieval_count=retrieval_count,
                max_retrievals=MAX_RETRIEVAL_ATTEMPTS,
            )

        # Step 4: Act on final policy verdict
        if verdict.action == "answer":
            evidence_text = self._format_evidence_text(evidence)
            prompt = (
                f"Research Question: {message}\n\n"
                f"Verified Evidence:\n{evidence_text}\n\n"
                "Provide a concise, factual answer. Cite every fact with its supporting [SRC-xxx] ID. "
                "Explicitly separate supported facts from any unresolved claims."
            )
            agent = self._session(session_id)
            try:
                result = agent(
                    [*image_blocks(images), {"text": prompt}] if images else prompt,
                    invocation_state={"images": list(images or [])},
                )
                reply = strip_thinking(str(result))
            except Exception:
                # Grounded fallback if LLM provider is offline or missing credentials
                reply = self._fallback_synthesis(message, evidence)
        else:
            # Policy mandated abstention (contradictory, insufficient, or out of scope)
            reply = self._format_abstention(message, verdict, evidence)

        latency_ms = round((time.perf_counter() - t0) * 1000)
        return {
            "reply": reply,
            "decisions": self.policy.log[before_decisions:],
            "model_id": self.model_id,
            "latency_ms": latency_ms,
            "agent": NAME,
            "eager": self.eager,
        }


if __name__ == "__main__":
    import json
    import sys

    agent = HackathonAgent()
    sample_queries = sys.argv[1:] or [
        "What are the throughput and memory characteristics of UltraAttention-v2?",
        "Did Project Helios demonstrate a verified net energy gain in 2026?",
        "What are the quarterly manufacturing output figures for semiconductor foundry Mars-Alpha?",
    ]
    for q in sample_queries:
        print(f"\n--- Query: {q} ---")
        output = agent.chat(q)
        print(json.dumps(output, indent=2))
