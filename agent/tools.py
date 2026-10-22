"""SourceSignal AI tools: local evidence corpus and retrieval functions.

Provides access to a curated, transparent technical evidence corpus with verified source IDs.
Every fact returned is grounded in an explicit source record. Never fabricate citations.
"""

from __future__ import annotations

import re
from typing import Any

from strands import tool

EVIDENCE_CORPUS: dict[str, dict[str, Any]] = {
    "SRC-101": {
        "title": "UltraAttention-v2 Architecture and Scaling Benchmark",
        "source": "Journal of Machine Learning Systems (March 2026)",
        "text": (
            "UltraAttention-v2 achieves 1M token context processing by utilizing structured block-sparse matrix kernels. "
            "On standard synthetic retrieval and long-document reasoning benchmarks, it sustains 2,400 tokens/second "
            "throughput while reducing active KV-cache GPU memory footprints by 68% compared to standard flash-attention baselines."
        ),
        "metadata": {
            "published": "2026-03-15",
            "reliability": "high",
            "domain": "AI Architecture",
            "peer_reviewed": True,
        },
    },
    "SRC-102": {
        "title": "Thermal Bottlenecks and Sustained Load in UltraAttention Implementations",
        "source": "Hardware Systems Quarterly (May 2026)",
        "text": (
            "Under prolonged continuous inference workloads exceeding 500k context lengths, UltraAttention-v2 exhibits "
            "severe memory fragmentation and thermal throttling on PCIe-based accelerator cards. Measurements show "
            "effective sustained throughput drops by 42% unless active liquid cooling and custom SRAM buffer management are deployed."
        ),
        "metadata": {
            "published": "2026-05-10",
            "reliability": "high",
            "domain": "Hardware Systems",
            "peer_reviewed": True,
        },
    },
    "SRC-201": {
        "title": "Project Helios Pulsed Stellarator Net Energy Gain Claim",
        "source": "Helios Energy Lab Technical Report (February 2026)",
        "text": (
            "Project Helios reported achieving Q_plasma of 1.25 in its pulsed stellarator reactor in February 2026. "
            "The team reported a measured net electrical output of 14.2 MJ from an 11.3 MJ magnetic compression pulse "
            "over a 400ms steady-state confinement window."
        ),
        "metadata": {
            "published": "2026-02-18",
            "reliability": "medium",
            "domain": "Nuclear Fusion",
            "peer_reviewed": False,
        },
    },
    "SRC-202": {
        "title": "Independent Audit and Calorimetry Calibration of Project Helios",
        "source": "International Fusion Verification Council (April 2026)",
        "text": (
            "An independent physical audit of Project Helios found systematic sensor drift and calibration flaws in "
            "diagnostic thermal sensors. Recalculated neutron spectroscopy and heat balance data proved actual energy "
            "output was 10.1 MJ against 11.3 MJ input (Q_plasma = 0.89), directly disproving claims of net energy gain."
        ),
        "metadata": {
            "published": "2026-04-22",
            "reliability": "high",
            "domain": "Nuclear Fusion",
            "peer_reviewed": True,
        },
    },
    "SRC-301": {
        "title": "Strands Decider 2B: Calibrated Decision Architecture",
        "source": "Strands AI Technical Whitepaper (January 2026)",
        "text": (
            "Strands Decider 2B is a 2-billion parameter decision model based on Qwen3.5. Unlike generative models that "
            "produce unstructured free-form text tokens, Strands Decider evaluates input state and typed questions (noul, "
            "choice, score) in a single forward pass, outputting calibrated posterior probabilities directly usable for "
            "programmatic policy enforcement and workflow routing."
        ),
        "metadata": {
            "published": "2026-01-20",
            "reliability": "high",
            "domain": "Decision Models",
            "peer_reviewed": True,
        },
    },
    "SRC-302": {
        "title": "Edge and Private Inference Efficiency with Strands Decider",
        "source": "Madurai AI Systems Review (February 2026)",
        "text": (
            "Benchmark evaluations confirm Strands Decider 2B executes inference in under 250ms on Apple Silicon (MLX) "
            "and dedicated GPU instances. Because decisions do not generate autoregressive token sequences, inference cost "
            "is approximately $0.035 per 1,000 decisions with deterministic latency and zero hallucination of structural control flow."
        ),
        "metadata": {
            "published": "2026-02-28",
            "reliability": "high",
            "domain": "Edge AI",
            "peer_reviewed": True,
        },
    },
    "SRC-401": {
        "title": "Nordic Hyperscale Center Power Usage Effectiveness (PUE) 2025-2026",
        "source": "Cloud Sustainability Index (January 2026)",
        "text": (
            "Nordic hyperscale data centers achieved an annualized Power Usage Effectiveness (PUE) of 1.12 during 2025-2026. "
            "This efficiency was driven by ambient free-air cooling and direct-to-chip heat recovery networks exported to "
            "municipal district heating infrastructure."
        ),
        "metadata": {
            "published": "2026-01-14",
            "reliability": "high",
            "domain": "Green Computing",
            "peer_reviewed": True,
        },
    },
    "SRC-501": {
        "title": "Solid-State LLZO Electrolyte Degradation and Cycle Life",
        "source": "Advanced Materials Energy Letters (November 2025)",
        "text": (
            "Lithium-lanthanum-zirconium-oxide (LLZO) solid electrolytes demonstrate less than 3% capacity fade over 1,200 "
            "rapid-charge cycles at 25 degrees Celsius when paired with single-crystal NMC811 cathode formulations."
        ),
        "metadata": {
            "published": "2025-11-30",
            "reliability": "high",
            "domain": "Energy Storage",
            "peer_reviewed": True,
        },
    },
}


STOP_WORDS = {
    "what", "are", "the", "for", "in", "and", "is", "of", "to", "did", "from",
    "with", "how", "does", "can", "an", "a", "this", "that", "these", "those",
    "or", "on", "at", "by", "about", "it", "its", "into", "as", "per", "were",
}


def search_corpus(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Search the local corpus for relevant sources based on keyword and token matching."""
    tokens = [w for w in re.findall(r"\b[A-Za-z0-9_-]{2,}\b", query.lower()) if w not in STOP_WORDS]
    if not tokens:
        tokens = re.findall(r"\b[A-Za-z0-9_-]{2,}\b", query.lower())
    if not tokens:
        return []

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for sid, data in EVIDENCE_CORPUS.items():
        title_words = set(re.findall(r"\b[A-Za-z0-9_-]{2,}\b", data["title"].lower()))
        text_words = set(re.findall(r"\b[A-Za-z0-9_-]{2,}\b", data["text"].lower()))
        domain_words = set(re.findall(r"\b[A-Za-z0-9_-]{2,}\b", data["metadata"].get("domain", "").lower()))
        all_words = title_words | text_words | domain_words

        matches = sum(1 for w in tokens if w in all_words)
        title_matches = sum(1 for w in tokens if w in title_words)
        score = (title_matches * 3.0) + matches

        # Direct source ID match (e.g. searching "SRC-101")
        if sid.lower() in query.lower():
            score += 15.0

        # Require meaningful match score
        if score >= 2.0 or sid.lower() in query.lower():
            scored.append((score, sid, data))

    scored.sort(key=lambda x: x[0], reverse=True)
    results: list[dict[str, Any]] = []
    for _, sid, data in scored[:limit]:
        results.append({
            "source_id": sid,
            "title": data["title"],
            "source": data["source"],
            "text": data["text"],
            "metadata": data["metadata"],
        })
    return results


@tool
def query_evidence(query: str) -> dict[str, Any]:
    """Retrieve verified evidence snippets and source IDs from the SourceSignal corpus for a query."""
    results = search_corpus(query, limit=3)
    if not results:
        return {
            "query": query,
            "results_count": 0,
            "results": [],
            "message": f"No relevant evidence found in the verified corpus for '{query}'.",
        }
    return {
        "query": query,
        "results_count": len(results),
        "results": results,
    }


@tool
def fetch_source(source_id: str) -> dict[str, Any]:
    """Fetch the full record and metadata for a specific source ID (e.g., 'SRC-101')."""
    sid = source_id.strip().upper()
    item = EVIDENCE_CORPUS.get(sid)
    if not item:
        return {"error": f"Source '{source_id}' does not exist in the verified corpus."}
    return {
        "source_id": sid,
        "title": item["title"],
        "source": item["source"],
        "text": item["text"],
        "metadata": item["metadata"],
    }


TOOLS = [query_evidence, fetch_source]
