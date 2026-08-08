# J Street Knowledge Assistant — POC

A retrieval-augmented Q&A assistant that answers questions about J Street using only J Street's own public site content (mission/about, endorsement criteria, policy FAQ, policy center experts, staff, polling, press, careers, current campaigns, and 2024 financials).

This is a scoped-down adaptation of Assist², an earlier RAG system I built with hybrid dense + BM25 retrieval, cross-encoder reranking, RAGAS evaluation, and a MongoDB Atlas backend. None of that complexity is appropriate for a 72-hour POC, so this version strips it down to the core pipeline: chunk → embed → retrieve → generate, with citations.

## How it works

1. **Ingestion (`ingest.py`)** — reads every `.txt`/`.pdf` file in `jstreet_corpus/`, splits each into paragraph-based chunks (~1000 characters, with a bit of overlap so a fact near a paragraph boundary doesn't get orphaned), and loads them into a local Chroma collection. This only needs to run once, or again after the corpus changes.
2. **Embeddings** — Chroma's default embedding model (`all-MiniLM-L6-v2`, running locally via ONNX) turns each chunk into a vector at ingest time and turns each question into a vector at query time. No embedding API calls, no API key needed for this step.
3. **Retrieval** — on each question, the app does a cosine-similarity search over the chunk vectors and pulls back the top 4 most relevant chunks.
4. **Generation (`app.py`)** — the retrieved chunks are numbered and passed to Claude (`claude-opus-5`) as context, with a system prompt instructing it to answer *only* from that context and cite chunk numbers. The answer streams into a Streamlit chat UI, with the raw retrieved chunks shown underneath in an expander so you can verify the citation against the actual source text.

## Why these tools

- **Chroma over MongoDB Atlas** — Atlas is what Assist² uses in production, but standing up a hosted vector DB is pure infrastructure overhead for a demo that needs to run on one laptop in under a minute. Chroma is an embedded, zero-setup vector store — `pip install` and go.
- **Local embeddings over an embedding API** — avoids a second API dependency and API key just to turn text into vectors, and it's free. Quality is lower than something like OpenAI's or Voyage's embedding models, but for a ~180-chunk corpus the difference doesn't matter.
- **Claude for generation** — grounded answer generation with citation instructions is exactly what a Q&A assistant needs, and it's easy to reason about and demo.
- **Streamlit over a bare CLI or notebook** — this needs to be watchable in a 5–10 minute video. A CLI loop would work but a browser UI is easier to follow on screen, and Streamlit needed almost no code to get there.
- **Dense-only retrieval, no reranking** — Assist² uses BM25 + dense hybrid retrieval with cross-encoder reranking. For a ~180-chunk corpus, plain dense similarity search is already precise enough (see the FAQ example below) and reranking would add complexity with no visible benefit at this scale.

## Limitations / what I'd improve with more time

1. **No hybrid retrieval or reranking.** Dense-only search works well here because the corpus is small and topically narrow, but it can miss exact-keyword matches (e.g. a specific bill name or dollar figure) that BM25 would catch. Assist²'s hybrid + cross-encoder approach exists specifically to fix this at scale.
2. **No evaluation suite.** There's no RAGAS or similar automated check on answer faithfulness or retrieval precision — I'm eyeballing quality on a handful of test questions rather than measuring it. For anything beyond a demo I'd want a small labeled eval set.
3. **No citation verification or hallucination guardrails beyond prompting.** The system prompt tells Claude to answer only from context and cite sources, but nothing programmatically checks that the cited chunk actually supports the claim. A real deployment would want at least a lightweight verification pass.
4. **Small, manually-curated corpus and no access control.** ~180 chunks from 9 files, hand-copied because `jstreet.org/robots.txt` disallows automated crawling on the relevant paths (respected here — no scraper was run against the live site). A production internal tool would need a much larger corpus, a real content-refresh pipeline, and authentication/access controls, none of which are in scope for a POC.

## Running it

```bash
pip install -r requirements.txt
python ingest.py          # builds chroma_db/ from jstreet_corpus/
export ANTHROPIC_API_KEY=your-key-here
streamlit run app.py
```

## Suggested walkthrough (5–10 min)

1. **Intro (30s)** — what this is, that it's adapted from Assist² and scoped down for the take-home, and that it runs entirely on J Street's own public content.
2. **Architecture (1–2 min)** — quick walk through ingest → embed → retrieve → generate, pointing at `ingest.py` and `app.py`.
3. **Demo query 1 — general** (e.g. "What is J Street's mission?") — show a clean grounded answer with citations.
4. **Demo query 2 — precision test** (e.g. "Does J Street support cutting US aid to Israel?" vs. a follow-up "What about Iron Dome funding specifically?") — show that retrieval pulls the *correct*, *different* chunks for two closely related but distinct policy positions, and expand the "Retrieved sources" panel to prove it.
5. **Demo query 3 — off-corpus question** (something not covered, e.g. a question about a different organization) — show the assistant declining rather than hallucinating.
6. **Wrap-up (1 min)** — walk through the limitations above and what you'd do with more time.
