# Contributing to ClinVerify

Thanks for contributing. This project follows a spec-driven workflow: design
documents and ADRs are the source of truth, and every change must pass the same
gates CI runs.

## 1. Setup

```bash
# Install uv (https://docs.astral.sh/uv/) then:
uv sync
cp .env.example .env        # optional: set OPENAI_API_KEY for generation
```

Python 3.11+ is required (CI uses 3.11; local dev commonly uses 3.12).

## 2. Quality Gates

Run the full gate set before opening a PR:

```bash
uv run ruff format .            # apply formatting
uv run ruff check .             # lint
uv run ruff format --check .    # verify formatting
uv run mypy                     # strict typecheck
uv run pytest                   # unit + deterministic eval checks
```

CI blocks on any failure, plus the Ragas evaluation gate (see below).

## 3. Test Tiers

| Tier | Command | Notes |
| :-- | :-- | :-- |
| Unit + deterministic eval | `uv run pytest` | Default; no network/models |
| Integration | `CLINVERIFY_INTEGRATION=1 uv run pytest` | Needs Qdrant + model downloads |
| Ragas gate | `CLINVERIFY_EVAL=1 uv run pytest -m eval` | Calls a real LLM; needs `OPENAI_API_KEY` |

Integration tests expect Qdrant on `localhost:6333`:

```bash
docker run -d --name clinverify-qdrant -p 6333:6333 qdrant/qdrant:latest
```

See `docs/testing.md` for the full strategy and coverage requirements.

## 4. Workflow

1. Read the relevant `docs/` spec and `docs/adr/` decision before changing
   behavior.
2. Keep changes scoped to one phase/feature; update tests alongside code.
3. Update `README.md` and `CHANGELOG.md` for user-visible changes.
4. Record significant decisions as a new ADR in `docs/adr/` (next sequential
   number) using the existing template.

## 5. Coding Standards

- **No comments** unless they explain non-obvious intent.
- Strict typing: mypy `strict=true`; avoid `Any` and add `# type: ignore[...]`
  only with a specific error code.
- Line length 100; ruff lint set `E,F,I,UP,B,SIM`.
- Prefer dependency injection via protocols (`Embedder`, `VectorStore`, chat
  client) so units are testable with fakes — no real models in unit tests.

## 6. Commits & PRs

- Small, focused commits with an imperative subject (e.g.
  `feat(eval): implement Phase 5 Ragas evaluation gate`).
- Never commit secrets, `.env`, or large artifacts.
- Ensure the Ragas gate passes (or note why the golden dataset is unaffected).

## 7. Failure-Mode Coverage

Every failure mode in `docs/sdd.md` §5 must have at least one test or a
documented deferral (tracked in `docs/testing.md` §8). When you add or change a
mitigation, update that table.
