# ADR-0007: Containerized microservices via Docker Compose

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [architecture.md §1](../architecture.md), [product.md §5](../product.md), [sdd.md §2.1](../sdd.md)

## Context

The system comprises a FastAPI service, a stateful Qdrant vector store, and a Streamlit UI, plus local ML model inference. It must be reproducible for developers, CI, and reviewers, and deployable with minimal operational overhead. A stated non-functional target is a single `docker compose up` deployment footprint.

## Decision

Package the system as **three containers** orchestrated by **Docker Compose**:
- `api` — FastAPI + Uvicorn (owns ingestion, retrieval, rerank, generation, audit).
- `qdrant` — vector store with a persistent volume.
- `ui` — Streamlit inspector that talks only to the API.

The UI must never connect to Qdrant directly, preserving a single service boundary.

## Consequences

### Positive
- One-command local bring-up matches the deployment footprint SLA.
- Clear service boundaries and independent scaling/upgrade paths.
- Reproducible across dev, CI, and review environments.

### Negative / Trade-offs
- Model weights inflate the `api` image; cold-start and image size need management.
- Compose is single-host; production orchestration (K8s) is a future concern, not addressed here.
- Network hop between `ui` and `api` adds minor latency vs. an in-process UI.

## Alternatives Considered

- **Monolith (UI + API + store in one process)** — rejected: no clean boundary; stateful vector store hard to manage.
- **Kubernetes from day one** — rejected: over-engineered for current scale and team size.
- **Serverless (Lambda) for the API** — rejected: model cold-starts and stateful vector store don't fit the model.
