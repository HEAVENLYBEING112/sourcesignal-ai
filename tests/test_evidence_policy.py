"""Tests for SourceSignal AI evidence policy, retrieval tools, and research agent."""

from __future__ import annotations

import re
from fakes import FakeDecider

from agent.agent import HackathonAgent, MAX_RETRIEVAL_ATTEMPTS
from agent.decision import EvidencePolicyCheck, QUESTIONS
from agent.tools import EVIDENCE_CORPUS, fetch_source, query_evidence, search_corpus


def _fake_rule(sufficient: float, contradictory: float, action: str):
    def rule(state, name, q):
        if name == "evidence_sufficient":
            return sufficient
        if name == "contradictory":
            return contradictory
        if name == "evidence_action":
            return action
        return 0.5
    return rule


def test_sufficient_evidence_allows_answer():
    """When relevant, sufficient evidence is present without conflict, the policy approves answering."""
    decider = FakeDecider(_fake_rule(sufficient=0.95, contradictory=0.05, action="answer"))
    policy = EvidencePolicyCheck(decider)
    evidence = [
        {"source_id": "SRC-101", "title": "UltraAttention Benchmark", "text": "UltraAttention sustains 2,400 tokens/sec."}
    ]
    verdict = policy.check("What is UltraAttention throughput?", evidence, retrieval_count=1, max_retrievals=2)
    assert verdict.action == "answer"
    assert verdict.evidence_sufficient >= 0.5
    assert verdict.contradictory < 0.4
    assert len(policy.log) == 1
    assert policy.log[0]["pattern"] == "#policy"


def test_contradictory_evidence_forces_abstain():
    """Contradictory claims between sources force abstention even if facts were retrieved."""
    decider = FakeDecider(_fake_rule(sufficient=0.8, contradictory=0.9, action="abstain"))
    policy = EvidencePolicyCheck(decider)
    evidence = [
        {"source_id": "SRC-201", "title": "Helios Claim", "text": "Net electricity output was 14.2 MJ."},
        {"source_id": "SRC-202", "title": "Helios Audit", "text": "Net output was 10.1 MJ (Q=0.89), disproving net gain."},
    ]
    verdict = policy.check("Did Helios achieve net gain?", evidence, retrieval_count=1, max_retrievals=2)
    assert verdict.action == "abstain"
    assert verdict.contradictory >= 0.4
    assert "Contradictory evidence" in verdict.reason


def test_missing_evidence_triggers_retrieval_then_abstain():
    """Missing or insufficient evidence triggers retrieval on first attempt, then abstains at boundary."""
    decider = FakeDecider(_fake_rule(sufficient=0.1, contradictory=0.05, action="retrieve"))
    policy = EvidencePolicyCheck(decider)

    # Attempt 1: triggers another retrieval attempt
    v1 = policy.check("What is deep sea cobalt yield?", [], retrieval_count=1, max_retrievals=2)
    assert v1.action == "retrieve"

    # Attempt 2: at boundary (retrieval_count == max_retrievals), policy must abstain
    v2 = policy.check("What is deep sea cobalt yield?", [], retrieval_count=2, max_retrievals=2)
    assert v2.action == "abstain"
    assert "Insufficient evidence" in v2.reason


def test_retrieval_loop_bounded():
    """HackathonAgent never exceeds MAX_RETRIEVAL_ATTEMPTS in its decision log."""
    decider = FakeDecider(_fake_rule(sufficient=0.1, contradictory=0.05, action="retrieve"))
    agent = HackathonAgent(decider=decider)
    res = agent.chat("Query with no relevant evidence")
    assert len(res["decisions"]) <= MAX_RETRIEVAL_ATTEMPTS
    assert res["decisions"][-1]["action"] == "abstain"
    assert "ABSTAIN" in res["reply"]


def test_citation_grounding_in_answers():
    """Answers generated from sufficient evidence cite verified [SRC-xxx] tags from the corpus."""
    decider = FakeDecider(_fake_rule(sufficient=0.95, contradictory=0.05, action="answer"))
    agent = HackathonAgent(decider=decider)
    res = agent.chat("What are the scaling and memory characteristics of UltraAttention-v2?")
    assert res["decisions"][0]["action"] == "answer"
    cited = re.findall(r"\[(SRC-\d+)\]", res["reply"])
    assert len(cited) >= 1
    for cid in cited:
        assert cid in EVIDENCE_CORPUS, f"Fabricated source ID {cid} found in answer!"


def test_contradiction_abstention_explanation():
    """Abstention for contradictory claims explains the conflicting sources explicitly."""
    decider = FakeDecider(_fake_rule(sufficient=0.6, contradictory=0.85, action="abstain"))
    agent = HackathonAgent(decider=decider)
    res = agent.chat("Did Project Helios verify net energy gain in its 2026 stellarator trials?")
    assert res["decisions"][0]["action"] == "abstain"
    assert "Contradictory Evidence" in res["reply"]
    # Check that both conflicting sources are highlighted
    assert "SRC-201" in res["reply"]
    assert "SRC-202" in res["reply"]


def test_evidence_tools_functionality():
    """Test search_corpus, query_evidence, and fetch_source tools."""
    # Search returns valid results
    results = search_corpus("UltraAttention", limit=2)
    assert len(results) >= 1
    assert results[0]["source_id"] in ("SRC-101", "SRC-102")

    # query_evidence tool format
    tool_res = query_evidence("PUE Nordic data center")
    assert tool_res["results_count"] >= 1
    assert tool_res["results"][0]["source_id"] == "SRC-401"

    # fetch_source tool
    src_res = fetch_source("SRC-301")
    assert src_res["title"] == "Strands Decider 2B: Calibrated Decision Architecture"

    # fetch_source missing
    err_res = fetch_source("SRC-999")
    assert "error" in err_res
