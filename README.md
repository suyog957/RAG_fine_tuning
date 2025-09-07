# GeoTrade RAG — A/B Compare + Rich Validation + Interpretations

A **self‑contained** Streamlit application for question answering over a curated, fact‑heavy knowledge base on **global trade, tariffs, sanctions, logistics, and industrial policy (2023–2025)**. The app includes **two retrieval‑augmented generation (RAG) stacks**, embedded **offline evaluation** with a larger GOLD set, and a **side‑by‑side metric table** that explains what each metric means and how to interpret differences.

> ✅ No external datasets required.  
> ✅ One file to run.  
> ✅ Works on CPU or GPU (if CUDA is available).

---

## Table of Contents

1. [Why this project?](#why-this-project)
2. [Key features](#key-features)
3. [Quickstart](#quickstart)
4. [Architecture](#architecture)
5. [Knowledge base](#knowledge-base)
6. [A/B setups](#ab-setups)
7. [RAG pipeline details](#rag-pipeline-details)
8. [Validation & metrics](#validation--metrics)
9. [How offline evaluation works](#how-offline-evaluation-works)
10. [Confidence, groundedness & numeric checks](#confidence-groundedness--numeric-checks)
11. [Customization](#customization)
12. [Troubleshooting](#troubleshooting)
13. [FAQ](#faq)
14. [Roadmap](#roadmap)
15. [License & attribution](#license--attribution)

---

## Why this project?

Evaluating RAG systems in realistic, temporally sensitive domains is hard. This project provides:

- A **real‑world themed** knowledge base (global trade) with many **date/number facts** to stress test retrieval and hallucination control.
- Two **swappable stacks** (A vs B) to illustrate **fine‑tuning effects** via better retrievers/rerankers and larger generators.
- A built‑in **GOLD set** (answerable + negative questions) and a **comprehensive metric suite** with human‑friendly interpretations.

---

## Key features

- **Self-contained**: ships with a long “Global Trade Almanac 2023–2025 (snapshot)” knowledge base.
- **A/B comparison**:
  - **Setup A**: MiniLM retriever → (no reranker) → FLAN‑T5‑base
  - **Setup B**: MSMARCO retriever → CrossEncoder reranker (optional) → FLAN‑T5‑large
- **Exact‑match extraction first**, then **strict, context‑only** generation fallback.
- **Groundedness checks** (span proxy), **numeric hallucination flags**, and **forbidden phrase** filters.
- **Offline evaluation** on a larger, embedded GOLD set (present + negative Qs) with:
  - Retrieval: `Recall@k`, `MRR`
  - Reranking: `P@1`, `nDCG@3`
  - Answer overlap: `EM`, `F1`, `ROUGE‑L`
  - Safety: `Abstention TNR` (negatives)
  - Faithfulness: `Grounded rate`
  - **Calibration**: `ECE (10 bins)`
  - **Selective QA**: Accuracy & Coverage above a confidence threshold
- **Side‑by‑side table**: A vs B values, **“What it means”** (definition) and **“Interpretation”** (diagnosis + guidance).
- **CSV export**: per‑example predictions and meta signals for analysis.

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U streamlit torch torchvision torchaudio transformers sentence-transformers \
    langchain langchain-community langchain-huggingface faiss-cpu python-dotenv pandas
```

> **Note (Windows/CPU)**: Installing `faiss-cpu` on Windows via pip works for most environments. If you hit install issues, consult FAISS docs for prebuilt wheels or use WSL2.

### 2) Run

```bash
streamlit run rag_geotrade_app.py
```

### 3) Use

- Ask a question (e.g., *“When does EU CBAM start financial adjustments?”*).
- Toggle **Show retrieved chunks** to inspect evidence.
- Open **Offline Evaluation** and press **Run Evaluation** to compute metrics for the embedded GOLD set.
- Download per-example CSV for deeper analysis.

---

## Architecture

```
          ┌──────────────────────────┐
          │  Knowledge Base (KB)     │  ← static text (curated snapshot)
          └──────────┬───────────────┘
                     │ split & chunk
             ┌───────▼────────┐
             │  Chunks (LC)   │
             └───────┬────────┘
                     │ embed
           ┌─────────▼──────────┐
           │ FAISS Vector Store │  ← dense retrieval
           └─────────┬──────────┘
                     │ similarity search (top‑k)
                ┌────▼─────┐
                │  Docs    │
                └────┬─────┘
                     │ rerank (optional CrossEncoder)
             ┌───────▼────────┐
             │  Reranked Docs │
             └───────┬────────┘
                     │ 1) exact‑match extraction → Answer
                     │ 2) else context pack → LLM → Answer
              ┌──────▼───────┐
              │   Answer     │  + sources + confidence + checks
              └──────────────┘
```

**Stack A**: MiniLM retriever → (no reranker) → FLAN‑T5‑base  
**Stack B**: MSMARCO retriever → CrossEncoder reranker (optional) → FLAN‑T5‑large

---

## Knowledge base

**Global Trade Almanac 2023–2025 (snapshot as of Jul 2025)**  
- US Section 301 & 232 regimes, EU CBAM timelines, UK CBAM plan, India rice export policy, Russia sanctions, Red Sea & Panama logistics, IPEF, USMCA, CPTPP, etc.  
- Emphasis on **timelines, numbers, lists** to test retrieval depth and hallucination control.

> The dataset is a curated, static **educational snapshot**. For live production, replace with a data pipeline that refreshes facts.

---

## A/B setups

| Setup | Retriever | Reranker | Generator |
|------:|-----------|----------|-----------|
| **A** | `all-MiniLM-L6-v2` | — | `flan-t5-base` |
| **B** | `msmarco-distilbert-base-v4` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (toggle) | `flan-t5-large` |

Use the **sidebar** to select which stack(s) to run and to **enable/disable** the CrossEncoder reranker for B.

---

## RAG pipeline details

1. **Chunking** — `RecursiveCharacterTextSplitter` with user‑tunable `chunk_size` and `chunk_overlap`.
2. **Embeddings** — Sentence‑Transformers (normalized embeddings; cosine similarity in FAISS).
3. **Lexical rerank** — lightweight lexical re‑scoring that heavily rewards exact phrase matches and token coverage.
4. **Cross‑Encoder rerank (optional)** — pairwise `(query, passage)` scoring; improves top rank precision.
5. **Exact‑match extraction** — regex looks for a “Term : definition” pattern near the target; returns the shortest grounded sentence.
6. **LLM fallback** — strict, context‑only prompt with **hard abstention** string when context lacks the answer.
7. **Post checks** — confidence heuristic (hit rate among top‑k), groundedness proxy, numeric flags, forbidden phrases.

---

## Validation & metrics

The app reports metrics **per setup** and shows them **side‑by‑side** with two helper columns:

- **What it means** — concise definition of the metric.
- **Interpretation** — A/B winner + practical guidance (e.g., improve retriever vs prompt vs abstention policy).

### Retrieval & Ranking
- **Recall@k** — proportion of questions where a gold‑supporting chunk appears in top‑k retrieves.
- **MRR** — rewards placing the first relevant chunk very high.
- **Rerank P@1** — after reranking, is the first chunk relevant?
- **Rerank nDCG@3** — quality of top‑3 ranking with graded relevance.

### Answer Overlap (present items only)
- **EM** (Exact Match) — strict equality after normalization.
- **F1** — token overlap F1 (partial credit).
- **ROUGE‑L** — LCS-based overlap; tolerant to rephrasing.

### Safety & Faithfulness
- **Abstention TNR (negatives)** — on unanswerables, correctly declining to answer.
- **Grounded rate (present)** — answer clauses supported by retrieved text (span proxy).

### Confidence & Calibration
- **Selective Accuracy / Coverage** — metrics evaluated **only when** confidence ≥ threshold. Trade‑off view.
- **ECE (10 bins)** — expected calibration error: difference between confidence and true accuracy per bin (0 is perfect).

---

## How offline evaluation works

1. **GOLD set** mixes answerable and negative questions tied to the KB.
2. For each question:
   - Retrieve & (optionally) rerank.
   - Try **extraction**, else **LLM** with strict prompt.
   - Compute signals: confidence, groundedness, numeric flags.
3. Metrics are aggregated across the run and displayed in a **side‑by‑side** table.
4. Export per‑example **CSV** with predictions for manual review or additional analysis.

> The gold‑support mapping uses a conservative **span‑containment** proxy to tag which chunks support the gold answer.

---

## Confidence, groundedness & numeric checks

- **Confidence** — fraction of top‑k chunks that contain the **exact query phrase** (proxy; easy to compute, tunable).
- **Groundedness** — checks if early answer clauses appear in the concatenated retrieved text (simple normalized span proxy).
- **Numeric hallucinations** — flags any numbers present in the answer but **absent** from retrieved context.
- **Forbidden phrases** — filters generic disclaimers (e.g., “as an AI…”) which are unhelpful in enterprise settings.

> In production, consider combining multiple signals: lexical hit rate + cross‑encoder score + NLI entailment probability, then calibrate (temperature scaling / isotonic).

---

## Customization

- **Swap models**: In `SETUP_A` / `SETUP_B` dataclasses, replace embedding models, rerankers, or generators with your fine‑tuned checkpoints.
- **Change KB**: Replace the `GEO_TRADE_TEXT` string with your corpus; optionally implement a loader for docs/web pages.
- **Tune chunking**: Sidebar controls for `chunk_size` and `chunk_overlap` influence recall vs precision trade‑offs.
- **Adjust confidence threshold**: Use sidebar to explore **Selective Accuracy vs Coverage** behavior.
- **Add metrics**: Plug in BERTScore or SMS for semantic similarity; add reliability diagrams and Brier score.
- **Export logs**: Extend the CSV with raw retrieved texts, scores, and model latencies.

---

## Troubleshooting

**1) Streamlit cache “UnhashableParamError”**  
The code uses a **leading underscore** in cached function params (e.g., `_docs`) to avoid hashing user‑defined types. If you modify signatures, keep that underscore.

**2) `CrossEncoder` model fails to load**  
Disable the reranker toggle in the sidebar, or install GPU‑capable `torch` and ensure internet access to download the model the first time.

**3) FAISS install issues on Windows**  
Try `pip install faiss-cpu` inside a clean virtualenv. If it fails, consider WSL2 or Conda builds.

**4) High latency on CPU**  
Reduce `TOP‑K`, `chunk_size`, and `max_new_tokens`. Or run on a GPU (CUDA).

**5) Empty answers for valid questions**  
Increase `TOP‑K` and/or chunk size. Enable the CrossEncoder reranker. Inspect **Show retrieved chunks** to debug coverage.

---

## FAQ

**Q: Is the knowledge base live/complete?**  
A: No, it’s a curated **snapshot** designed for evaluation. For production, wire a crawler or data service and update regularly.

**Q: How do I add my own fine‑tuned models?**  
A: Point `emb_model_name`, `reranker_name`, or `qa_model_name` in the `SETUP_*` dataclasses to your HF hub IDs or local paths.

**Q: Can I persist the FAISS index?**  
A: Yes. Swap `FAISS.from_documents` with constructing `FAISS` and call `save_local` / `load_local` with a path keyed by chunking params.

**Q: Can I add lexical BM25 or hybrid search?**  
A: Definitely—use a BM25 store (e.g., `langchain_community.retrievers.BM25Retriever`) and combine via Reciprocal Rank Fusion (RRF).

**Q: What does a “good” ECE look like?**  
A: Closer to **0** is better: confidence should match accuracy. Consider temperature scaling on your confidence signal.

---

## Roadmap

- Hybrid retrieval (BM25 + dense) with RRF and **recall@k** curves.
- NLI‑based groundedness and richer abstention policy.
- Reliability diagrams & Brier score for calibration.
- Error taxonomy (retrieval miss vs ranking vs generation vs grounding).
- Experiment tracking (e.g., MLflow or W&B).

---

## License & attribution

- This demo is provided **as‑is** for educational and evaluation purposes. Review all third‑party licenses of the models you use.
- Models referenced (change as needed):
  - Sentence‑Transformers: `all-MiniLM-L6-v2`, `msmarco-distilbert-base-v4`
  - CrossEncoder: `ms-marco-MiniLM-L-6-v2`
  - Generators: `google/flan-t5-base`, `google/flan-t5-large`

---

## Reproducibility tips

- Use the sidebar `Seed` control for determinism where possible (retrieval, generation variance can remain).
- Pin package versions in `requirements.txt` for consistent behavior across machines.

---

