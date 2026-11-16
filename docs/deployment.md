# Deployment

ClinVerify ships as a single `docker compose up` stack (ADR-0007): the FastAPI
service, a persistent Qdrant store, and the Streamlit inspector.

## 1. Topology

```
        ┌──────────┐   HTTP    ┌──────────┐   gRPC/HTTP   ┌──────────┐
        │   ui     │ ────────► │   api    │ ────────────► │  qdrant  │
        │ Streamlit│           │ FastAPI  │               │ (volume) │
        └──────────┘           └────┬─────┘               └──────────┘
                                    │ OpenAI API (generation)
                                    └──────────► api.openai.com
```

The UI connects **only** to the API; it never reaches Qdrant directly.

## 2. Prerequisites

- Docker Engine with Compose v2.
- An `OPENAI_API_KEY` for generation (ingest and retrieval work without it).
- Network access on first run to download ONNX/transformer weights.

## 3. Bring-Up

```bash
# Optional: provide the LLM key (otherwise generation returns 500 upstream_llm_error)
export OPENAI_API_KEY=sk-...

docker compose up --build
```

| Service | URL |
| :-- | :-- |
| API (docs) | http://localhost:8000/docs |
| Streamlit inspector | http://localhost:8501 |
| Qdrant dashboard | http://localhost:6333/dashboard |

First startup downloads embedding and reranker models into the `model_cache`
volume; subsequent starts are fast and offline-capable for retrieval.

## 4. Environment Variables

| Variable | Default | Purpose |
| :-- | :-- | :-- |
| `OPENAI_API_KEY` | – | Generation; unset → `upstream_llm_error` on `/query` |
| `QDRANT_URL` | `http://localhost:6333` (`http://qdrant:6333` in Compose) | Vector store endpoint |
| `QDRANT_API_KEY` | – | Optional store auth |
| `QDRANT_COLLECTION` | `clinical_trials_hybrid` | Collection name |
| `API_BASE_URL` | `http://localhost:8000` (`http://api:8000` in Compose) | UI → API endpoint |
| `MAX_UPLOAD_MB` | `50` | Upload size cap (FM-1) |
| `MAX_PAGES` | `300` | Page-count guard (FM-1) |
| `EMBED_BATCH_SIZE` | `64` | Embedding/upsert batch size (FM-1) |
| `GENERATOR_MODEL` | `gpt-4o-mini` | Generation model |
| `GENERATOR_TEMPERATURE` | `0.0` | Deterministic grounding (FM-3) |
| `PREFETCH_K` / `RERANK_TOP_K` / `RRF_K` | `20` / `5` / `60` | Retrieval hyperparameters |

See `.env.example` for the full list.

## 5. Volumes

| Volume | Mount | Contents |
| :-- | :-- | :-- |
| `qdrant_storage` | `/qdrant/storage` | Indexed vectors and payloads |
| `model_cache` | `/models` | Hugging Face (`HF_HOME`) and FastEmbed caches |

Both survive `docker compose down`; use `docker compose down -v` to reset.

## 6. Health Checks

| Service | Probe | Gate |
| :-- | :-- | :-- |
| `qdrant` | TCP `127.0.0.1:6333` | `api` waits for `service_healthy` |
| `api` | `GET /health` | `ui` waits for `service_healthy` |
| `ui` | `GET /_stcore/health` | – |

## 7. Walkthrough (exit-criteria smoke test)

```bash
# 1. Ingest a PDF (retrieval works without an API key)
curl -s -F "file=@label.pdf" \
  "http://localhost:8000/ingest?doc_id=FDA_Apixaban_2025"

# 2. Query
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the apixaban dose?", "mode": "hybrid_rerank", "top_k": 5}'

# 3. Inspect in the UI (answer, citation-audit status, telemetry, source drawers)
open http://localhost:8501
```

A verified answer returns `citation_verified: true`; insufficient evidence returns
the `INSUFFICIENT_CONTEXT` abstention with `abstained: true`.

## 8. Operations Notes

- **Single host:** Compose is single-node. Multi-node/Kubernetes orchestration is
  a future concern (ADR-0007).
- **Scaling:** the API is stateless apart from the model cache; scale it
  horizontally behind a load balancer with a shared Qdrant.
- **Upgrades:** rebuild the image (`docker compose up --build`); the lockfile
  guarantees reproducible dependencies.
- **Observability:** API logs are single-line JSON on stdout — ship them to your
  aggregator (`docs/observability.md`).
- **First query warms the reranker:** the cross-encoder loads on first
  `hybrid_rerank` query; its weights are cached in `model_cache`, so subsequent
  queries are fast. Pre-warm by issuing one query after deploy.
- **Known limitation:** retrieval and generation run synchronously inside async
  endpoints, so a long model load can briefly block the event loop. Run the API
  with multiple workers (`--workers N`) or move inference to a threadpool for
  high-concurrency deployments.
