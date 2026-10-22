0. SourceSignal AI is an evidence-gated research agent powered by Amazon Bedrock. Strands Decider evaluates evidence sufficiency and contradiction policy before any answer is generated.

1. An evidence-gated research agent that validates evidence sufficiency and detects contradictions with Strands Decider before answering. What it does: Retrieve verified facts and citations from a curated evidence corpus. Evaluate evidence sufficiency, relevance, and contradictions with Strands Decider 2B. Abstain transparently when evidence is contradictory or missing. Run on Amazon Bedrock AgentCore Runtime paired with Amazon Bedrock.

2. The agent you are about to see is running on Amazon Bedrock AgentCore Runtime.

3. The agent retrieves supporting benchmarks from SRC-101 and the policy approves answering with verified citations. The decider scored evidence sufficient 0.91, contradictory 0.32, confidence 0.65, so the verdict was answer, in 3.5 seconds.

4. The agent detects contradictory findings between SRC-201 and SRC-202, so the policy mandates abstention. The decider scored evidence sufficient 0.95, contradictory 0.86, confidence 0.63, so the verdict was abstain, in 3.7 seconds.

5. No relevant evidence exists in the corpus, so the policy prevents hallucination and abstains. The decider scored evidence sufficient 0.17, contradictory 0.30, confidence 0.84, so the verdict was retrieve, in 2.2 seconds. The decider scored evidence sufficient 0.18, contradictory 0.30, confidence 0.55, so the verdict was abstain, in 2.2 seconds.

6. On 39 labelled decisions, with the decider on the AWS User Group Madurai hosted API, Strands Decider was 79% accurate at a median of 3684 milliseconds, against 77% and 22355 milliseconds for a hosted open LLM asked the same questions.

7. One forward pass per decision with calibrated confidence, zero hallucinated citations, and Amazon Bedrock for grounded synthesis.