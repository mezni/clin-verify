# ADR-0008: Ragas evaluation gate in CI

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [sdd.md §6](../sdd.md), [architecture.md §2.4](../architecture.md), [product.md §5](../product.md)

## Context

Retrieval and generation quality degrade silently when models, prompts, chunking, or fusion parameters change. In a regulated domain, quality regressions must be caught before merge, not discovered in production. Unit tests verify mechanics but not answer quality.

## Decision

Add a **Ragas-based evaluation gate** to CI (GitHub Actions), run on every pull request against a curated **golden dataset** (`golden_dataset.json`) of pharmacological Q&A pairs:
- `Faithfulness ≥ 0.95`
- `LLMContextPrecisionWithReference ≥ 0.88`

A threshold breach **blocks merge**. The golden dataset is version-controlled alongside the code.

## Consequences

### Positive
- Objective, automated regression detection for retrieval/generation changes.
- Quantifies the impact of model, prompt, and chunking changes before they ship.
- Golden dataset doubles as living documentation of expected behavior.

### Negative / Trade-offs
- Ragas evaluation calls an LLM, adding CI cost and runtime; may need caching or a smaller gating subset.
- LLM-based metrics are themselves non-deterministic — thresholds need tolerance and periodic re-baselining.
- Golden dataset coverage must be actively maintained or the gate becomes a false signal.

## Alternatives Considered

- **No automated quality gate** — rejected: silent regressions unacceptable.
- **Exact-match assertions only** — rejected: too brittle for free-text clinical answers.
- **Human review per PR** — rejected: unscalable and inconsistent.
- **Custom LLM-judge harness** — deferred: Ragas provides standard, comparable metrics out of the box.
