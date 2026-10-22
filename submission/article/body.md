# SourceSignal AI: Building an Evidence-Gated Research Agent with Strands Decider 2B

By Surya

**What if a research agent knew when not to answer?** SourceSignal AI is an evidence-gated research agent that uses Strands Decider 2B to decide whether to answer, retrieve more evidence, or abstain. In our held-out evaluation, the Decider workflow achieved 79.5% decision accuracy versus 76.9% for the LLM-only baseline, with a median latency of 3.684 seconds versus 22.355 seconds.

## The project

Research agents can produce plausible answers even when the available evidence is incomplete, irrelevant, or contradictory. SourceSignal AI explores a different approach: evidence must justify an answer before the agent responds.

The project is designed for research workflows where traceability and uncertainty matter. It retrieves records from a labelled evidence corpus, evaluates evidence sufficiency and contradictions, and chooses the next action under an explicit policy.

Its main features are:

- Evidence retrieval using source identifiers and query matching.
- A decision stage powered by Strands Decider 2B.
- Three possible actions: `answer`, `retrieve`, and `abstain`.
- Source-cited answers when evidence supports a claim.
- A retrieval limit of two attempts to prevent unbounded searching.
- A labelled evaluation suite with 26 cases: 13 tuning cases and 13 held-out test cases.
- A comparison against an LLM-only baseline.

A live example demonstrates the difference between unsupported and supported questions.

**Question:** "How does UltraAttention-v2 reduce active KV-cache GPU memory usage?"

**Agent response:** UltraAttention-v2 reduces active KV-cache GPU memory footprints by 68% compared with standard flash-attention baselines by utilizing structured block-sparse matrix kernels. The response cites `[SRC-101]`.

The agent selected `answer` after one retrieval attempt. In contrast, when asked about a return policy using the same source, it abstained because that source documents model architecture and performance rather than product returns.

That distinction is central to SourceSignal AI: a question is not answerable merely because the model can generate a response.

## How I built it

SourceSignal AI uses Strands Agents to coordinate the research workflow and expose retrieval tools. The tools query the local evidence corpus and fetch complete records by source ID. Strands Decider 2B evaluates the evidence-policy questions and guides the next action.

Amazon Bedrock provides the generative model, while Amazon Bedrock AgentCore Runtime hosts the deployed agent. The deployed runtime was tested successfully in `us-east-1` using Amazon Nova Pro for generation.

The implementation keeps the decision policy explicit rather than hiding every choice inside a general-purpose language-model prompt. The agent receives a question, retrieves evidence, evaluates the evidence, and either answers with citations or abstains.

The project was developed with the AWS User Group Madurai Strands Decider hackathon kit. Its spec-driven structure, labelled evaluation cases, and automated checks helped keep the implementation aligned with the required reusable pattern and measurable comparison.

## The decision I handed to Strands Decider 2B

The most important design choice was deciding whether the available evidence justified answering a research question.

This decision needs to be fast because it sits in the agent's execution path. It also needs explicit options so the agent can distinguish missing evidence from contradictory evidence instead of treating every uncertainty as the same problem.

The implementation in `agent/decision.py` defines three typed questions:

- **`evidence_sufficient`** - Does the collected evidence contain relevant and sufficient factual information to answer the research query? The options are `true` and `false`.
- **`contradictory`** - Do the evidence sources present conflicting or mutually contradictory claims? The options are `true` and `false`.
- **`evidence_action`** - What action should be taken next? The options are `answer`, `retrieve`, and `abstain`.

The action descriptions encode the policy. `answer` means sufficient and consistent evidence is available for a grounded, cited response. `retrieve` means another search may find the missing facts. `abstain` means the evidence is absent, irrelevant, or irreconcilably contradictory.

The agent allows at most two retrieval attempts. If evidence remains insufficient after the permitted attempts, it abstains rather than inventing supporting facts. The recorded decisions include the selected action, evidence-sufficiency and contradiction signals, confidence, reasoning, retrieval count, and decision latency.

## Measured

I evaluated the workflow using a labelled set of 26 cases, with 13 held out for testing. The held-out split produced 39 recorded decisions. The following figures are from the project's measured comparison.

| Held-out test metric | Strands Decider 2B workflow | LLM-only baseline |
|---|---:|---:|
| Decision accuracy | 79.5% | 76.9% |
| Median latency | 3.684 seconds | 22.355 seconds |
| 95th-percentile latency | 4.224 seconds | 49.805 seconds |
| Cost per 1,000 decisions | $0.035 | $0.000 |

The results show a modest improvement in held-out decision accuracy and approximately 6.1 times lower median latency for the Decider workflow. The measured Decider cost was $0.035 per 1,000 decisions, compared with $0.000 for the baseline under the evaluation's accounting assumptions.

The Decider workflow is not better on every metric. In the held-out evaluation, the baseline answered more cases at the chosen confidence threshold, while the Decider workflow was more accurate among its confident decisions. That trade-off matters: a system that abstains more often may reduce unsupported answers but can also leave useful questions unanswered.

These results describe this particular evaluation suite, not every research workload. The corpus and test set are limited, and broader testing is needed to establish how well the policy generalizes.

The Decider was called through the configured Strands Decider service, while the deployed generative workflow used Amazon Bedrock and ran on Amazon Bedrock AgentCore Runtime. The live test of the UltraAttention-v2 question returned a cited answer in approximately 6.1 seconds end to end.

## The reusable decision pattern

The reusable pattern is labelled `#policy` and is implemented in `agent/decision.py`.

The pattern separates the decision policy from the rest of the research workflow. Another builder can adapt the approach by defining typed questions for evidence sufficiency and contradiction, providing explicit answer/retrieve/abstain options, and connecting each decision to a clear action in their own agent.

The important part is not simply adding a model call. It is making the policy observable and testable: decisions are recorded, retrieval is bounded, evidence is cited, and abstention is an intentional outcome.

For a different domain, the evidence questions and action descriptions should be adapted to that domain's requirements. A medical research agent, for example, would need its own evidence sources, validation strategy, and safety constraints rather than reusing this project's corpus unchanged.

## What I learned and what is next

The main lesson is that answer quality depends on the decision to answer as much as on the ability to generate fluent text. Explicit evidence checks made it possible to test when the agent should answer and when it should stop.

I also learned that latency, decision accuracy, confidence, and coverage need to be evaluated together. A system can improve accuracy among confident decisions while answering fewer questions, so reporting only one metric would hide an important trade-off.

The current implementation uses a small labelled corpus, and retrieval quality depends on the sources available to it. Next, I would expand the corpus, add more difficult contradiction cases, evaluate citation precision, and test across more diverse research questions. I would also investigate how confidence thresholds affect answer coverage and when human review should be requested.

SourceSignal AI is a working demonstration of an evidence-gated agent deployed on Amazon Bedrock AgentCore. Its central principle is straightforward: a trustworthy research agent should know not only how to answer, but when the available evidence is not enough.

**Code:** Add the public GitHub repository URL after publishing the repository.

**Video:** Add the public YouTube or Vimeo demo URL after uploading the recording.

**Tags:** #strands-decider-2b #policy

Built with the AWS User Group Madurai Strands Decider hackathon kit.