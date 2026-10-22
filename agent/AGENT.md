# Use case: SourceSignal AI — Evidence-Gated Research Agent

- **Pitch:** SourceSignal AI is an evidence-gated technical research agent that answers complex questions using a transparent evidence corpus, citing verifiable source IDs and using Strands Decider 2B to enforce strict evidence policy (answer, retrieve, or abstain).
- **Who it is for:** Technical researchers, domain analysts, and engineering teams who need dependable, hallucination-free technical answers backed by transparent evidence snippets without fabricated citations.
- **Features:**
  - Local evidence retrieval tool querying verified source IDs, titles, snippets, and metadata;
  - Evidence policy classifier (`#policy`) evaluating whether evidence is relevant, sufficient, contradictory, and trustworthy;
  - Programmatic action gating: answers only when evidence is sufficient and consistent, triggers bounded retrieval attempts when partial, and abstains when evidence conflicts or is missing;
  - Zero citation hallucination: all factual claims must cite supporting `[SRC-xxx]` identifiers;
  - Contradiction detection: conflicting empirical reports are surfaced neutrally rather than picking an arbitrary winner;
  - Deployed on Amazon Bedrock AgentCore Runtime.
- **The decision handed to Strands Decider 2B:** Under `#policy`, evaluates whether retrieved evidence is sufficient and non-contradictory, and decides the next action: `answer`, `retrieve`, or `abstain`.
- **Why a decision model:** The policy check runs on every retrieval turn, so it must be fast (under 250ms), inexpensive (~$0.035 per 1,000 decisions), and private. Most importantly, it returns calibrated probabilities that Python code can branch on with deterministic control flow, eliminating generative hallucination of routing logic.
- **Amazon Bedrock:** Generates the final grounded synthesis from verified evidence snippets (Amazon Nova Pro by default).
