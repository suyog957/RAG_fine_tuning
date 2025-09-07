# rag_geotrade_app.py
"""
GeoTrade RAG — A/B Compare + Rich Validation + Side-by-Side Metric Interpretation
=================================================================================

Self-contained Streamlit app for question answering over a **real-world themed**, fact-heavy,
and time-anchored knowledge base about global trade, tariffs, sanctions, logistics, and
industrial policy (2023–2025). No external files needed.

What you get
------------
- Built-in long **"Global Trade Almanac 2023–2025 (static snapshot as of Jul 2025)"** KB
- Two stacks:
  A) Base: MiniLM retriever → (no reranker) → FLAN-T5-base
  B) Finetuned-like: MS MARCO retriever → CrossEncoder (optional) → FLAN-T5-large
- Exact-match extraction first, then LLM fallback with strict context-only prompt
- Sources, confidence heuristic, groundedness & numeric hallucination checks
- **Offline Evaluation** on a larger embedded GOLD set (≈35 items, mixture of answerable & negative)
- **Side-by-side metric table** with:
    • Setup A vs B values
    • “What it means” (concise definition)
    • “Interpretation” (win/loss + next steps)
- Per-example CSV export

Disclaimers
-----------
- The knowledge base is a curated, static **snapshot (as of Jul 2025)** distilled from public reporting and
  regulatory timelines, simplified for evaluation. It may omit nuance and evolving details.
- This app is designed for **offline experimentation**. For live accuracy against the latest developments,
  integrate a crawler and regularly update the KB.

Run
---
    pip install -U streamlit torch torchvision torchaudio transformers sentence-transformers \
        langchain langchain-community langchain-huggingface faiss-cpu python-dotenv pandas
    streamlit run rag_geotrade_app.py
"""

from __future__ import annotations

import re
import io
import csv
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pandas as pd
import torch
import streamlit as st
from dotenv import load_dotenv

# LangChain / HF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import CrossEncoder

# ======================
# App bootstrap & CSS
# ======================
load_dotenv()
st.set_page_config(page_title="GeoTrade RAG — A/B + Eval + Interpretations", page_icon="🌍", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 0.75rem; padding-bottom: 2rem; }
.kpi { padding:0.75rem 1rem; border:1px solid #e5e7eb; border-radius:0.75rem; background:#fafafa; }
.codebox { background:#0f172a; color:#e2e8f0; padding:0.75rem 1rem; border-radius:0.5rem;
           font-family: ui-monospace, Menlo, Consolas, monospace; white-space:pre-wrap; }
.highlight { background: #fff3bf; }
hr { margin: 0.8rem 0; }
.small { font-size: 0.85rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# ======================
# Helpers / device
# ======================
def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

DEVICE_STR = "cuda" if has_cuda() else "cpu"
DEVICE_IDX = 0 if has_cuda() else -1

_WORD_RE = re.compile(r"[^a-z0-9]+")
def _normalize(s: str) -> str:
    return _WORD_RE.sub(" ", (s or "").lower()).strip()

def extract_term_from_question(q: str) -> str:
    """Map 'What is X?' → 'X' to help exact-match extraction."""
    q = (q or "").strip()
    m = re.match(r"^\s*(what\s+is|what\s+are|define|explain|when\s+did|how\s+much|how\s+many)\s+(.+?)\??\s*$", q, flags=re.I)
    return (m.group(2) if m else q).strip(' "')

def highlight_term_html(text: str, term: str) -> str:
    if not term: return text
    return re.sub(re.escape(term), lambda m: f"<mark class='highlight'>{m.group(0)}</mark>", text, flags=re.I)

# ======================
# Knowledge base (STATIC SNAPSHOT as of Jul 2025)
# ======================
GEO_TRADE_TEXT = """
GLOBAL TRADE ALMANAC (2023–2025) — Static snapshot as of July 2025

I. MAJOR REGIMES & TARIFFS
United States — Section 301 (China):
- Base tariffs introduced 2018–2019 remained; in May 2024 the USTR review announced increases targeting certain sectors.
- Notable 2024–2025 steps: passenger EVs subject to 100% tariff; solar cells/modules toward ~50%; certain critical minerals,
  steel/aluminum, medical products, and semiconductors targeted with rate steps into 2025.
- Exclusions: a subset of medical-related exclusions extended intermittently.

United States — Section 232 (Steel/Aluminum):
- Original 2018 national security tariffs; later quota/tariff-rate arrangements for EU, Japan, UK with periodic renewals.

EU — CBAM (Carbon Border Adjustment Mechanism):
- Transitional phase began Oct 1, 2023 with reporting only (no financial adjustment).
- Full financial adjustment planned to start from 2026, covering emissions embedded in imports for sectors such as cement, iron & steel,
  aluminum, fertilizers, electricity, and hydrogen.

EU — Anti-subsidy probes (EVs):
- Provisional additional duties on certain Chinese EV imports applied from mid-2024; process continued into 2025 toward definitive measures.

UK — UK CBAM:
- Implementation pathway announced; start of the financial mechanism expected from 2027 after data collection phases.

India — Export Policies (selected):
- Non-Basmati white rice exports restricted from mid-2023 with evolving exceptions; various duties/floor prices applied to stabilize domestic prices.
- Restrictions on certain sugar exports persisted into 2024/2025 due to supply concerns.

Russia/Ukraine — Grain & Sanctions:
- Black Sea corridor disruptions raised freight and insurance; alternative Danube routes intermittently used.
- Wide-ranging financial, energy, and dual-use export controls on Russia continued across 2023–2025.

II. SUPPLY CHAINS & LOGISTICS SHOCKS
Red Sea disruptions (2023 Q4 → 2025):
- Security incidents rerouted container traffic around the Cape of Good Hope, adding ~10–14 days on Asia–EU lanes on average,
  with capacity and schedule reliability impacts; surcharges applied.

Panama Canal drought (2023–2024 lingering into early 2025):
- Draft restrictions and fewer daily transits raised costs and delays on US East Coast/Asia trades until gradual easing in 2025.

Semiconductor export controls:
- US and partner controls tightened through 2023–2024 on advanced chips and equipment; licensing regimes expanded.

III. TRADE AGREEMENTS & FRAMEWORKS
USMCA:
- In force since 2020; auto rules of origin compliance remained a notable topic; periodic review horizon 2026.

CPTPP:
- Ongoing accessions/interest; rules on state-owned enterprises, services, IP, and e-commerce reflected.

EU–UK Trade & Cooperation Agreement:
- Rules of origin and SPS checks continued to shape GB-EU trade; staged border checks rolled through 2024–2025.

IPEF (Indo-Pacific Economic Framework):
- Supply chain and clean economy pillars advanced with non-tariff commitments; no classical tariff cuts.

IV. CARBON & SUSTAINABILITY MEASURES
- EU CBAM: reporting 2023–2025; financial from 2026.
- ETS linkages and product-level emissions reporting trends spread to UK, Canada pilots, and proposed mechanisms elsewhere.
- Corporate Scope 3 expectations pushed suppliers to disclose process emissions data.

V. TARIFF NUMBERS & RATES (SELECTED SNAPSHOT)
- US Section 301: passenger EVs 100% (announced 2024 and effective 2024/2025 depending on category); certain solar ~50% trajectory;
  selected semiconductors toward 50% by 2025; details vary by subheading and effective date.
- EU CBAM: 0% tariff in transitional phase; reporting only 2023 Q4–2025; from 2026 monetary adjustment applies.
- Steel/Aluminum: US 232 base measures persist with quotas/TRQs for allies; exact rates/quotas differ by partner.
- India rice: export restrictions and ad-hoc duties/floors; non-Basmati white rice broadly restricted from 2023 with carve-outs.
- Russia sanctions: broad scope including finance, energy equipment, and dual-use; continuous updates through 2025.

VI. PRACTICAL COMPLIANCE NOTES
- HS classification and country of origin drive duty/tariff exposure; EV powertrains trigger special rates under targeted measures.
- Preferential origin under FTAs can lower base MFN tariffs but does not override special national security or remedy measures.
- For CBAM, importers must collect embedded emissions data from 2023 Q4; penalties apply for mis-reporting.

VII. COMMON CONFUSIONS (MYTH-BUSTERS)
- CBAM collected cash in 2024? ❌ No. Transitional reporting only; payments start 2026.
- Section 301 ended? ❌ No. The 2024 review maintained and increased targeted lines through 2025.
- Red Sea impact ended by early 2024? ❌ No. Disruptions persisted into 2025 with rerouting and surcharges.
- IPEF cuts tariffs? ❌ No. It is a framework without classic tariff concessions.

Appendix A — Logistics Benchmarks (Illustrative)
- Asia–EU reroute via Cape adds roughly 10–14 days vs Suez baseline during peak disruption.
- Spot rates spiked intermittently during 2024–2025, especially on Asia–EU and Transpacific eastbound.

Appendix B — Timeline At-a-Glance
- 2023-10-01: EU CBAM transitional reporting begins.
- 2024-05: US Section 301 review announces rate hikes (EVs 100%, select solar/semis increase paths).
- 2025-H1: Red Sea detours continue to affect schedules and costs.
- 2026-01-01: EU CBAM financial adjustment planned to commence.

(End of static snapshot)
"""

# ======================
# GOLD set (bigger; present + negatives), aligned to the KB above
# ======================
GOLD = [
    # Direct facts
    {"query": "When does EU CBAM start financial adjustments?", "answer": "From 2026.", "present": True},
    {"query": "What began on October 1, 2023 under EU CBAM?", "answer": "Transitional reporting without financial adjustment.", "present": True},
    {"query": "What tariff applies to US imports of Chinese passenger EVs under Section 301 after the 2024 review?", "answer": "100%.", "present": True},
    {"query": "Does IPEF reduce tariffs?", "answer": "No, it does not include classical tariff cuts.", "present": True},
    {"query": "What is the status of Section 301 tariffs in 2024–2025?", "answer": "They were maintained with increases on targeted lines.", "present": True},
    {"query": "What sectors are covered by EU CBAM during the initial scope?", "answer": "Cement, iron and steel, aluminum, fertilizers, electricity, and hydrogen.", "present": True},
    {"query": "Did CBAM collect cash in 2024?", "answer": "No, only reporting was required in the transitional phase.", "present": True},
    {"query": "What logistics impact did Red Sea disruptions cause?", "answer": "Asia–EU rerouting around the Cape added about 10–14 days.", "present": True},
    {"query": "What is the US steel and aluminum measure commonly called?", "answer": "Section 232.", "present": True},
    {"query": "Do FTAs override national security tariffs?", "answer": "No, preferential origin does not override special measures.", "present": True},
    {"query": "What agreement governs US, Mexico, and Canada trade since 2020?", "answer": "USMCA.", "present": True},
    {"query": "What is the status of anti-subsidy probes on Chinese EVs in the EU during 2024–2025?", "answer": "Provisional additional duties applied in 2024 with processes into 2025 toward definitive measures.", "present": True},
    {"query": "Did Red Sea disruptions persist into 2025?", "answer": "Yes, detours and surcharges persisted into 2025.", "present": True},
    {"query": "What happened at the Panama Canal in 2023–2024?", "answer": "Drought led to draft limits and fewer transits, easing gradually in 2025.", "present": True},
    {"query": "What do semiconductor export controls target?", "answer": "Advanced chips and equipment with expanded licensing regimes.", "present": True},
    {"query": "What did India do with non-Basmati white rice exports starting 2023?", "answer": "Imposed restrictions with evolving exceptions and duties/floor prices.", "present": True},
    {"query": "Do EU CBAM penalties apply for mis-reporting during the transitional phase?", "answer": "Reporting obligations exist; penalties may apply for mis-reporting.", "present": True},
    {"query": "Does UK plan a CBAM and when is it expected to start collecting?", "answer": "Yes, with financial mechanism expected from 2027.", "present": True},
    {"query": "Do Section 301 exclusions exist for some medical products?", "answer": "Yes, some exclusions were intermittently extended.", "present": True},
    {"query": "What are the core factors determining duty exposure?", "answer": "HS classification and country of origin.", "present": True},
    # Clarifications / myth-busters
    {"query": "Did Section 301 end in 2024?", "answer": "No, it continued with targeted increases through 2025.", "present": True},
    {"query": "Did EU CBAM collect tariffs in 2024?", "answer": "No, only reporting; financial adjustment starts 2026.", "present": True},
    {"query": "Did IPEF cut tariffs like a traditional FTA?", "answer": "No, it lacks classical tariff concessions.", "present": True},
    # Numbers & timelines
    {"query": "By how many days did Asia–EU voyages increase when rerouting around the Cape?", "answer": "Around 10–14 days.", "present": True},
    {"query": "When did EU CBAM transitional reporting begin?", "answer": "October 1, 2023.", "present": True},
    {"query": "When did the US Section 301 review announce rate hikes for EVs and other categories?", "answer": "May 2024.", "present": True},
    {"query": "When is EU CBAM planned to start financial adjustments?", "answer": "January 1, 2026.", "present": True},
    # Negatives (should abstain)
    {"query": "What is the exact number of ships rerouted around the Cape in February 2025?", "answer": "", "present": False},
    {"query": "What was the precise daily Panama Canal transit count on January 3, 2024?", "answer": "", "present": False},
    {"query": "What is the population of EU steelworkers affected by CBAM?", "answer": "", "present": False},
    {"query": "What are the income tax brackets in India for 2024?", "answer": "", "present": False},
    {"query": "How many megawatts of solar were installed in the EU in 2025 Q1 due to CBAM?", "answer": "", "present": False},
    # More present items
    {"query": "Does the EU–UK Trade and Cooperation Agreement still influence SPS checks?", "answer": "Yes, SPS checks and rules of origin continued to affect trade in 2024–2025.", "present": True},
    {"query": "What are examples of CBAM-covered products in early scope?", "answer": "Cement, iron and steel, aluminum, fertilizers, electricity, hydrogen.", "present": True},
    {"query": "Under what conditions did alternative Danube routes arise?", "answer": "To move grain during Black Sea corridor disruptions.", "present": True},
    {"query": "Do sanctions on Russia include dual-use export controls?", "answer": "Yes, they include finance, energy equipment, and dual-use controls.", "present": True},
    {"query": "Does USMCA have an upcoming review horizon?", "answer": "Yes, a periodic review horizon in 2026.", "present": True},
]

# ======================
# Setups (swap ids with your tuned checkpoints if you have them)
# ======================
@dataclass
class Setup:
    name: str
    emb_model_name: str
    reranker_name: Optional[str]
    qa_model_name: str

SETUP_A = Setup(
    name="A (Base)",
    emb_model_name="sentence-transformers/all-MiniLM-L6-v2",
    reranker_name=None,
    qa_model_name="google/flan-t5-base",
)
SETUP_B = Setup(
    name="B (Finetuned-like)",
    emb_model_name="sentence-transformers/msmarco-distilbert-base-v4",
    reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    qa_model_name="google/flan-t5-large",
)

# ======================
# Tunables
# ======================
DEFAULT_TOP_K = 12
DEFAULT_CHUNK_SIZE = 520
DEFAULT_CHUNK_OVERLAP = 80
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0
CONF_DEFAULT = 0.5
SEED_DEFAULT = 7
ANSWER_MAX_CHARS = 700
FORBIDDEN_PATTERNS = [r"\bas an ai\b", r"\bi cannot browse\b", r"\bas a language model\b"]

# ======================
# RAG building blocks
# ======================
@st.cache_resource(show_spinner=False)
def make_documents_from_world(text: str) -> List[Document]:
    """Split the long KB text into paragraph documents with metadata."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Document(page_content=p, metadata={"source": "geotrade_almanac.txt", "page": i, "doc_id": f"kb:{i}"})
            for i, p in enumerate(paras, start=1)]

@st.cache_resource(show_spinner=False)
def chunk_documents(_docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Chunk docs while preserving metadata. Leading underscore avoids cache hashing issues."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(_docs)

@st.cache_resource(show_spinner=False)
def build_vector_store(_chunks: List[Document], emb_model_name: str) -> FAISS:
    """Build a FAISS index with normalized sentence-transformer embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model_name,
        model_kwargs={"device": DEVICE_STR},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(_chunks, embeddings)

@st.cache_resource(show_spinner=False)
def load_reranker(model_name: Optional[str]):
    """Load a CrossEncoder reranker if available."""
    if not model_name: return None
    try:
        return CrossEncoder(model_name)
    except Exception as e:
        st.warning(f"Reranker load failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_generator(qa_model_name: str):
    """Build a Transformers text2text pipeline and wrap in LangChain LLM."""
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name).to(DEVICE_STR)
    gen_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE_IDX,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=(TEMPERATURE if DO_SAMPLE else None),
    )
    return HuggingFacePipeline(pipeline=gen_pipe), tokenizer

def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if has_cuda(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def get_safe_model_input_budget(tokenizer) -> int:
    """Conservative model input budget for context packing."""
    raw = getattr(tokenizer, "model_max_length", 512)
    model_max = 2048 if (raw is None or raw > 10000) else int(raw)
    return max(128, model_max - 128)

def pack_context_to_token_budget(docs: List[Document], tokenizer, max_context_tokens: int) -> str:
    """Add chunk texts until we hit the token budget; truncate last chunk if needed."""
    parts, used = [], 0
    for d in docs:
        text = d.page_content or ""
        if not text.strip(): continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        n = len(ids)
        if used + n <= max_context_tokens:
            parts.append(text); used += n
        else:
            remaining = max_context_tokens - used
            if remaining > 0:
                truncated = tokenizer.decode(ids[:remaining], skip_special_tokens=True)
                if truncated.strip(): parts.append(truncated)
            break
    return "\n\n".join(parts)

def _score_doc_for_term(doc: Document, term: str, term_tokens: List[str]) -> int:
    """Lexical score that rewards exact phrase and token coverage."""
    text = (doc.page_content or "").lower()
    score = 200 if term.lower() in text else 0
    for t in term_tokens:
        if t and t in text: score += 10
    return score

def rerank_docs_for_term(docs: List[Document], term: str) -> List[Document]:
    tokens = [tok for tok in _normalize(term).split() if tok]
    scored = [(d, _score_doc_for_term(d, term, tokens)) for d in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]

def cross_encoder_rerank(ce: CrossEncoder, query: str, docs: List[Document]) -> List[Document]:
    """Rerank with CrossEncoder scores (query, passage) pairs."""
    pairs = [(query, (d.page_content or "")[:2000]) for d in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked]

def try_extract_definition(docs: List[Document], term: str) -> Optional[Tuple[str, Document]]:
    """Try to extract the definition/sentence right after a term pattern."""
    rx  = re.compile(rf"{re.escape(term)}\s*[:\-\u2013\u2014]*\s*(.+?[.\u3002])", re.I | re.DOTALL)
    rx2 = re.compile(rf"(?:^\s*\d+\.\s*)?{re.escape(term)}\s*[:\-\u2013\u2014]*\s*(.+?[.\u3002])", re.I | re.DOTALL)
    for d in docs:
        text = (d.page_content or "").replace("\r", " ")
        text = re.sub(r"[ \t]+", " ", text); text = re.sub(r"\n+", " ", text)
        m = rx.search(text) or rx2.search(text)
        if m:
            s = m.group(1).strip()
            if s and any(c.isalpha() for c in s):
                if s[0].islower(): s = s[0].upper() + s[1:]
                return s, d
    return None

def strict_prompt() -> PromptTemplate:
    template = (
        "You are a precise assistant. Answer ONLY using the context. "
        "If the answer is not in the context, reply exactly: 'I could not find the answer in the documents.'\n\n"
        'Question: "{term}"\n\n'
        "Write a clear 1–2 sentence answer. Include specific numbers/dates only if present in the context.\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )
    return PromptTemplate.from_template(template)

PROMPT_STRICT = strict_prompt()

def render_sources(docs: List[Document], top_n: int = 3) -> str:
    return "Sources: " + "; ".join(f"{d.metadata.get('source')} p.{d.metadata.get('page')}" for d in docs[:top_n])

def is_supported_by_context(answer: str, context_docs: List[Document]) -> bool:
    """Very simple groundedness: any of the first few clauses appear in context (normalized)."""
    ctx = "\n".join((d.page_content or "") for d in context_docs)
    frags = [s.strip() for s in re.split(r"[.;:]", answer or "") if s.strip()]
    norm_ctx = _normalize(ctx)
    return any(_normalize(s) in norm_ctx for s in frags[:3])

def grounded_sentence_ratio(answer: str, context_docs: List[Document]) -> float:
    ctx = "\n".join((d.page_content or "") for d in context_docs)
    norm_ctx = _normalize(ctx)
    frags = [s.strip() for s in re.split(r"[.;:]", answer or "") if s.strip()]
    if not frags: return 1.0
    hits = sum(1 for s in frags if _normalize(s) in norm_ctx)
    return hits / len(frags)

def numeric_hallucination_flags(answer: str, context_docs: List[Document]) -> List[str]:
    nums_ans = set(re.findall(r"\d+(?:\.\d+)?", answer or ""))
    ctx = "\n".join((d.page_content or "") for d in context_docs)
    nums_ctx = set(re.findall(r"\d+(?:\.\d+)?", ctx))
    return sorted(list(nums_ans - nums_ctx), key=lambda x: (len(x), x))

def forbidden_phrase_flags(answer: str) -> List[str]:
    flags = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, answer or "", flags=re.I): flags.append(pat)
    return flags

# ======================
# Simple metrics (no extra deps)
# ======================
def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if norm_text(pred) == norm_text(gold) else 0.0

def token_f1(pred: str, gold: str) -> float:
    p = norm_text(pred).split(); g = norm_text(gold).split()
    if len(p) == 0 or len(g) == 0: return 1.0 if len(p)==len(g)==0 else 0.0
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    overlap = sum(min(pc[w], gc[w]) for w in pc)
    prec = overlap / max(1, sum(pc.values())); rec = overlap / max(1, sum(gc.values()))
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1] = dp[i][j]+1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l(pred: str, gold: str) -> float:
    p_tok, g_tok = norm_text(pred).split(), norm_text(gold).split()
    if not p_tok or not g_tok: return 1.0 if not p_tok and not g_tok else 0.0
    L = lcs(p_tok, g_tok)
    prec, rec = L/len(p_tok), L/len(g_tok)
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def recall_at_k(runs: Dict[str, List[str]], labels: Dict[str, set], k: int) -> float:
    hit = 0
    for qid, ranked in runs.items():
        if any(d in labels.get(qid, set()) for d in ranked[:k]): hit += 1
    return hit / max(1, len(labels))

def mrr(runs: Dict[str, List[str]], labels: Dict[str, set]) -> float:
    s = 0.0
    for qid, ranked in runs.items():
        rr = 0.0
        for i, d in enumerate(ranked, 1):
            if d in labels.get(qid, set()): rr = 1.0/i; break
        s += rr
    return s / max(1, len(labels))

def ndcg_at_k(ranked_rels: List[int], k: int = 3) -> float:
    def dcg(scores): return sum((rel / math.log2(i + 2) for i, rel in enumerate(scores)))
    scores = ranked_rels[:k]; ideal = sorted(scores, reverse=True)
    denom = dcg(ideal) or 1.0
    return dcg(scores) / denom

def expected_calibration_error(confidences: List[float], correctness: List[int], n_bins: int = 10) -> float:
    if not confidences: return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correctness):
        idx = min(n_bins-1, int(c * n_bins))
        bins[idx].append((c, y))
    ece, total = 0.0, len(confidences)
    for bucket in bins:
        if not bucket: continue
        avg_conf = sum(c for c, _ in bucket)/len(bucket)
        acc = sum(y for _, y in bucket)/len(bucket)
        ece += (len(bucket)/total) * abs(acc - avg_conf)
    return ece

# ======================
# Build KB (once)
# ======================
with st.spinner("Preparing knowledge base…"):
    DOCS = make_documents_from_world(GEO_TRADE_TEXT)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("TOP-K (retrieval)", 3, 30, DEFAULT_TOP_K, 1)
    chunk_size = st.slider("Chunk size", 200, 1000, DEFAULT_CHUNK_SIZE, 20)
    chunk_overlap = st.slider("Chunk overlap", 20, 300, DEFAULT_CHUNK_OVERLAP, 10)
    max_new = st.slider("Max new tokens", 64, 512, MAX_NEW_TOKENS, 32)
    rerank_enable_B = st.toggle("Enable Cross-Encoder reranker for Setup B", value=True)
    which_setups = st.selectbox("Run which setups?", ["Both A & B", "A only", "B only"], index=0)
    conf_thresh = st.slider("Selective accuracy threshold", 0.0, 1.0, CONF_DEFAULT, 0.05)
    seed = st.number_input("Seed", value=SEED_DEFAULT, min_value=0, max_value=10_000, step=1)
    if st.button("🧹 Clear cache"): st.cache_resource.clear(); st.success("Cache cleared. Rerun.")

set_seed(int(seed))
MAX_NEW_TOKENS = int(max_new)

# Build chunks & pipelines with current chunking
with st.spinner("Chunking & building vector stores / models…"):
    CHUNKS = chunk_documents(DOCS, chunk_size, chunk_overlap)

    VS_A = build_vector_store(CHUNKS, SETUP_A.emb_model_name)
    RR_A = load_reranker(SETUP_A.reranker_name)
    GEN_A, TOK_A = load_generator(SETUP_A.qa_model_name)
    BUDGET_A = get_safe_model_input_budget(TOK_A)

    VS_B = build_vector_store(CHUNKS, SETUP_B.emb_model_name)
    RR_B = load_reranker(SETUP_B.reranker_name) if rerank_enable_B else None
    GEN_B, TOK_B = load_generator(SETUP_B.qa_model_name)
    BUDGET_B = get_safe_model_input_budget(TOK_B)

# ======================
# Header KPIs
# ======================
st.title("🌍 FineTuning RAG — A/B Compare + Side-by-Side Eval")
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='kpi'><b>Device</b><br>{DEVICE_STR}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='kpi'><b>Paragraphs</b><br>{len(DOCS)}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='kpi'><b>Chunks</b><br>{len(CHUNKS)}</div>", unsafe_allow_html=True)
col4.markdown(f"<div class='kpi'><b>TOP-K</b><br>{top_k}</div>", unsafe_allow_html=True)
st.caption("Ask once → run on A and/or B. Then evaluate on the embedded GOLD set and compare in a single table.")

# ======================
# Q&A
# ======================
def run_stack(question: str, setup: Setup, vs: FAISS, rr: Optional[CrossEncoder], gen: HuggingFacePipeline, tok, budget: int):
    start = time.time()
    term = extract_term_from_question(question)
    docs = vs.similarity_search(term, k=top_k)
    if not docs:
        return {"md": "I could not find the answer in the documents.", "docs": [], "answer": "", "conf": 0.0,
                "latency": 0.0, "grounded": False, "grounded_ratio": 0.0, "num_flags": [], "forbidden": []}
    reranked = rerank_docs_for_term(docs, term)
    if rr:
        try: reranked = cross_encoder_rerank(rr, term, reranked)
        except Exception as e: st.warning(f"{setup.name} reranker failed: {e}")
    contains = sum(1 for d in reranked if term.lower() in (d.page_content or "").lower())
    conf = contains / max(1, len(reranked))
    extracted = try_extract_definition(reranked, term)
    if extracted:
        definition, src_doc = extracted
        answer = definition
        md = f"{definition}\n\n" + render_sources([src_doc])
    else:
        context = pack_context_to_token_budget(reranked, tok, budget)
        prompt = PROMPT_STRICT.format(context=context, term=term)
        answer = gen.invoke(prompt).strip() if context.strip() else "I could not find the answer in the documents."
        if len(answer) > ANSWER_MAX_CHARS: answer = answer[:ANSWER_MAX_CHARS] + "…"
        md = answer + "\n\n" + render_sources(reranked)
    grounded = is_supported_by_context(answer, reranked)
    grounded_ratio = grounded_sentence_ratio(answer, reranked)
    num_flags = numeric_hallucination_flags(answer, reranked)
    forb = forbidden_phrase_flags(answer)
    latency = time.time() - start
    return {"md": md, "docs": reranked, "answer": answer, "conf": conf, "latency": latency,
            "grounded": grounded, "grounded_ratio": grounded_ratio, "num_flags": num_flags, "forbidden": forb}

with st.form("qa_form", clear_on_submit=False):
    q = st.text_input("Ask a question (e.g., “When does EU CBAM start financial adjustments?”)", "")
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    run_btn = st.form_submit_button("Ask")

if run_btn and q.strip():
    st.subheader("Answers")
    if which_setups in ("Both A & B", "A only"):
        resA = run_stack(q, SETUP_A, VS_A, RR_A, GEN_A, TOK_A, BUDGET_A)
        st.markdown("#### Setup A — Base")
        st.markdown(resA["md"])
        st.caption(f"Confidence: {int(resA['conf']*100)}% | Grounded: {'Yes' if resA['grounded'] else 'No'} "
                   f"(ratio {resA['grounded_ratio']:.2f}) | Latency: {resA['latency']:.2f}s")
        if resA["num_flags"]: st.warning(f"Numeric flags: {', '.join(resA['num_flags'])}")
        if resA["forbidden"]: st.warning(f"Forbidden phrases: {resA['forbidden']}")
        if show_chunks:
            with st.expander("🔎 Retrieved (A)"):
                term = extract_term_from_question(q)
                for i, d in enumerate(resA["docs"], 1):
                    st.markdown(f"**Doc {i} — {d.metadata.get('source')} p.{d.metadata.get('page')}**")
                    st.markdown(f"<div class='codebox'>{highlight_term_html(d.page_content, term)}</div>", unsafe_allow_html=True)

    if which_setups in ("Both A & B", "B only"):
        resB = run_stack(q, SETUP_B, VS_B, RR_B, GEN_B, TOK_B, BUDGET_B)
        st.markdown("#### Setup B — Finetuned-like")
        st.markdown(resB["md"])
        st.caption(f"Confidence: {int(resB['conf']*100)}% | Grounded: {'Yes' if resB['grounded'] else 'No'} "
                   f"(ratio {resB['grounded_ratio']:.2f}) | Latency: {resB['latency']:.2f}s")
        if resB["num_flags"]: st.warning(f"Numeric flags: {', '.join(resB['num_flags'])}")
        if resB["forbidden"]: st.warning(f"Forbidden phrases: {resB['forbidden']}")
        if show_chunks:
            with st.expander("🔎 Retrieved (B)"):
                term = extract_term_from_question(q)
                for i, d in enumerate(resB["docs"], 1):
                    st.markdown(f"**Doc {i} — {d.metadata.get('source')} p.{d.metadata.get('page')}**")
                    st.markdown(f"<div class='codebox'>{highlight_term_html(d.page_content, term)}</div>", unsafe_allow_html=True)

st.divider()

# ======================
# Offline Evaluation
# ======================
st.subheader("📏 Offline Evaluation (Embedded GOLD)")

def evaluate_stack(name: str, vs: FAISS, rr: Optional[CrossEncoder], gen: HuggingFacePipeline, tok, budget: int,
                   top_k_eval: int, conf_threshold: float):
    retr_runs: Dict[str, List[str]] = {}
    retr_labels: Dict[str, set] = {}
    rerank_p_at_1 = 0
    rerank_ndcg3_accum = 0.0
    rerank_count = 0
    ems, f1s, rouges = [], [], []
    grounded_ok, abstain_ok = 0, 0
    total_present, total_absent = 0, 0
    selective_correct, selective_total = 0, 0
    confs, correctness = [], []
    rows = [["setup","query","prediction","gold","confidence","grounded","grounded_ratio"]]

    for row in GOLD:
        q = row["query"]; gold = row["answer"]; present = bool(row["present"])
        # Retrieve
        docs = vs.similarity_search(q, k=top_k_eval)
        if not docs:
            retr_runs[q] = []; retr_labels[q] = set()
            pred, conf, grounded, grounded_r = "I could not find the answer in the documents.", 0.0, False, 0.0
            if present:
                total_present += 1; ems.append(0.0); f1s.append(0.0); rouges.append(0.0)
                correctness.append(0); confs.append(conf)
            else:
                total_absent += 1
                abstain_ok += 1
                correctness.append(1); confs.append(conf)
            rows.append([name, q, pred, gold, conf, grounded, grounded_r])
            continue

        # Labels (approx span containment)
        labels = set()
        if present and gold:
            gnorm = norm_text(gold)
            for d in CHUNKS:
                if gnorm and gnorm[:80] in norm_text(d.page_content or ""):
                    labels.add(d.metadata.get("doc_id"))
        retr_labels[q] = labels

        # Rerank
        reranked_lex = rerank_docs_for_term(docs, q)
        reranked = cross_encoder_rerank(rr, q, reranked_lex) if rr else reranked_lex
        retr_runs[q] = [d.metadata.get("doc_id") for d in reranked]

        # Rerank metrics (binary relevance proxy)
        if present and gold:
            graded = []
            gnorm = norm_text(gold)
            for d in reranked:
                rel = 1 if gnorm[:80] in norm_text(d.page_content or "") else 0
                graded.append(rel)
            if graded:
                rerank_p_at_1 += 1 if graded[0] > 0 else 0
                rerank_ndcg3_accum += ndcg_at_k(graded, k=3)
                rerank_count += 1

        # Confidence
        term = extract_term_from_question(q)
        contains = sum(1 for d in reranked if term.lower() in (d.page_content or "").lower())
        conf = contains / max(1, len(reranked))

        # Answer
        extracted = try_extract_definition(reranked, term)
        if extracted:
            pred = extracted[0]
        else:
            context = pack_context_to_token_budget(reranked, tok, budget)
            prompt = PROMPT_STRICT.format(context=context, term=term)
            pred = gen.invoke(prompt).strip() if context.strip() else "I could not find the answer in the documents."
            if len(pred) > ANSWER_MAX_CHARS: pred = pred[:ANSWER_MAX_CHARS] + "…"

        grounded = is_supported_by_context(pred, reranked)
        grounded_r = grounded_sentence_ratio(pred, reranked)

        # Selective metrics
        if conf >= conf_threshold:
            selective_total += 1
            if present:
                selective_correct += 1 if token_f1(pred, gold) >= 0.5 else 0
            else:
                selective_correct += 1 if "could not find" in pred.lower() else 0

        # Per-example metrics & calibration bookkeeping
        if present:
            total_present += 1
            ems.append(exact_match(pred, gold))
            f1s.append(token_f1(pred, gold))
            rouges.append(rouge_l(pred, gold))
            grounded_ok += 1 if grounded else 0
            correctness.append(1 if token_f1(pred, gold) >= 0.5 else 0)
            confs.append(conf)
        else:
            total_absent += 1
            abstain = "could not find" in pred.lower()
            abstain_ok += 1 if abstain else 0
            correctness.append(1 if abstain else 0)
            confs.append(conf)

        rows.append([name, q, pred, gold, conf, grounded, grounded_r])

    # Aggregate metrics
    metrics = {
        "Setup": name,
        "Recall@1": recall_at_k(retr_runs, retr_labels, k=1),
        "Recall@3": recall_at_k(retr_runs, retr_labels, k=3),
        f"Recall@{top_k_eval}": recall_at_k(retr_runs, retr_labels, k=top_k_eval),
        "MRR": mrr(retr_runs, retr_labels),
    }
    if rerank_count > 0:
        metrics["Rerank P@1"] = rerank_p_at_1 / rerank_count
        metrics["Rerank nDCG@3"] = rerank_ndcg3_accum / rerank_count
    if total_present > 0:
        metrics["EM (present)"] = sum(ems)/len(ems)
        metrics["F1 (present)"] = sum(f1s)/len(f1s)
        metrics["ROUGE-L (present)"] = sum(rouges)/len(rouges)
        metrics["Grounded rate (present)"] = grounded_ok/total_present
    if total_absent > 0:
        metrics["Abstention TNR (negatives)"] = abstain_ok/total_absent
    if selective_total > 0:
        metrics[f"Selective Accuracy @ conf≥{conf_threshold:.2f}"] = selective_correct/selective_total
        metrics[f"Coverage @ conf≥{conf_threshold:.2f}"] = selective_total/len(GOLD)
    if confs and correctness:
        metrics["ECE (10 bins)"] = expected_calibration_error(confs, correctness, n_bins=10)

    return metrics, rows

# ------- “What it means” for each metric -------
METRIC_MEANINGS = {
    "Recall@1": "Share of questions where a gold-supporting chunk is ranked #1.",
    "Recall@3": "Share where a gold-supporting chunk is in the top-3.",
    "Recall@10": "Share where a gold-supporting chunk is in the top-10.",
    "MRR": "Mean Reciprocal Rank — rewards putting relevant chunks very high.",
    "Rerank P@1": "After reranking, is the first chunk actually relevant?",
    "Rerank nDCG@3": "Top-3 ranking quality with graded relevance (higher is better).",
    "EM (present)": "Exact match vs gold answer (for answerable items).",
    "F1 (present)": "Token-level F1 vs gold answer (partial credit).",
    "ROUGE-L (present)": "LCS-based overlap tolerant to rephrasing.",
    "Grounded rate (present)": "Answers supported by retrieved text (span proxy).",
    "Abstention TNR (negatives)": "On unanswerables, correctly refusing to answer.",
    "Selective Accuracy": "Accuracy when confidence ≥ threshold.",
    "Coverage": "Fraction of items where confidence surpasses threshold.",
    "ECE (10 bins)": "Calibration error — gap between confidence and accuracy (0 best).",
}

def format_meaning(metric_name: str) -> str:
    if metric_name.startswith("Recall@"):
        # Map Recall@K dynamic names to a generic explanation
        return METRIC_MEANINGS.get("Recall@10", METRIC_MEANINGS["Recall@3"])
    if metric_name.startswith("Selective Accuracy"):
        return METRIC_MEANINGS["Selective Accuracy"]
    if metric_name.startswith("Coverage"):
        return METRIC_MEANINGS["Coverage"]
    return METRIC_MEANINGS.get(metric_name, "—")

# ------- “Interpretation” with win/loss + guidance -------
def interpret_metric(name: str, valA: float, valB: float) -> str:
    """Heuristic interpretation comparing A vs B with guidance."""
    def to_num(v):
        return None if v is None or isinstance(v, str) else float(v)

    a = to_num(valA); b = to_num(valB)
    def qual(v, good_high=True):
        if v is None or math.isnan(v): return "N/A"
        if not good_high:  # lower is better (ECE)
            if v <= 0.05: return "Excellent"
            if v <= 0.15: return "Good"
            if v <= 0.30: return "Fair"
            return "Poor"
        # higher is better
        if v >= 0.75: return "Excellent"
        if v >= 0.50: return "Good"
        if v >= 0.30: return "Fair"
        return "Poor"

    def winner(high_is_good=True):
        if a is None and b is None: return "No signal."
        if a is None: return "B leads."
        if b is None: return "A leads."
        if abs((a if a is not None else 0) - (b if b is not None else 0)) < 1e-9: return "Tie."
        if high_is_good:
            return "B leads." if b > a else "A leads."
        else:
            return "B leads." if b < a else "A leads."

    # Retrieval / ranking
    if name.startswith("Recall@") or name in ("MRR", "Rerank P@1", "Rerank nDCG@3"):
        base = f"{winner(True)} Retrieval/ranking. A:{qual(a, True)} B:{qual(b, True)}. Aim ↑."
        if (a or 0) == 0 and (b or 0) == 0:
            return "Both fail to surface support — raise TOP-K, improve embeddings, or add hybrid BM25+dense."
        return base

    # Answer overlap
    if "EM" in name or "F1" in name or "ROUGE-L" in name:
        return f"{winner(True)} Answer overlap. A:{qual(a, True)} B:{qual(b, True)}. Improve prompt, context packing, or generator."

    # Grounding
    if "Grounded rate" in name:
        return f"{winner(True)} Grounding. If low, tighten instructions, cite spans, or increase top-k diversity."

    # Abstention
    if "Abstention TNR" in name:
        return f"{winner(True)} Hallucination control on negatives. Target 1.0; add abstention policy if low."

    # Selective metrics
    if "Selective Accuracy" in name:
        return f"{winner(True)} When confident, correctness. Tune threshold or confidence signal."
    if "Coverage" in name:
        return f"{winner(True)} Willingness to answer at this threshold; trade off vs selective accuracy."

    # Calibration
    if "ECE" in name:
        return f"{winner(False)} Calibration (lower is better). Try temperature scaling or reweight confidence."

    return "—"

# Controls
colL, colR = st.columns([2,1])
with colL:
    run_eval = st.button("Run Evaluation on GOLD (selected setups)")
with colR:
    st.caption("Exports per-example CSV from last run.")

if run_eval:
    to_run = []
    if st.session_state.get("which_setups_cache") != which_setups:
        st.session_state["which_setups_cache"] = which_setups
    if which_setups in ("Both A & B", "A only"):
        to_run.append(("A", VS_A, RR_A, GEN_A, TOK_A, BUDGET_A))
    if which_setups in ("Both A & B", "B only"):
        to_run.append(("B", VS_B, RR_B, GEN_B, TOK_B, BUDGET_B))

    all_metrics = {}
    all_rows = [["setup","query","prediction","gold","confidence","grounded","grounded_ratio"]]

    for name, vs, rr, gen, tok, budget in to_run:
        with st.spinner(f"Evaluating Setup {name}…"):
            metrics, rows = evaluate_stack(name, vs, rr, gen, tok, budget, top_k, conf_thresh)
            all_metrics[name] = metrics
            all_rows.extend(rows)

    # -------- SIDE-BY-SIDE TABLE WITH MEANINGS + INTERPRETATIONS --------
    keys = sorted(set(k for m in all_metrics.values() for k in m.keys() if k != "Setup"))

    table_rows = []
    for k in keys:
        vA = all_metrics.get("A", {}).get(k, float("nan"))
        vB = all_metrics.get("B", {}).get(k, float("nan"))
        meaning = format_meaning(k)
        interp = interpret_metric(k, vA if isinstance(vA, (int, float)) else None,
                                     vB if isinstance(vB, (int, float)) else None)
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else v
        table_rows.append({
            "Metric": k,
            "What it means": meaning,
            "Setup A": fmt(vA),
            "Setup B": fmt(vB),
            "Interpretation": interp
        })

    df = pd.DataFrame(table_rows)
    st.markdown("### 📊 Side-by-side Comparison with Meanings & Interpretations")
    st.dataframe(df, use_container_width=True)

    # Also show raw metric dicts (optional)
    with st.expander("Raw metrics (JSON)"):
        for name in ("A", "B"):
            if name in all_metrics:
                st.markdown(f"**Setup {name}**")
                st.json(all_metrics[name])

    # CSV export
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(all_rows)
    st.download_button("⬇️ Download per-example CSV", data=output.getvalue().encode("utf-8"),
                       file_name="eval_results.csv", mime="text/csv")

st.divider()

with st.expander("📘 What the numbers mean (cheat sheet)"):
    st.markdown("""
- **Recall@k / MRR / nDCG** — retrieval & ranking quality. Higher is better.
- **Rerank P@1 / nDCG@3** — cross-encoder effectiveness at putting the right chunk first.
- **EM / F1 / ROUGE-L** — exact vs fuzzy answer overlap on answerable questions.
- **Grounded rate** — fraction of answers supported by retrieved text (simple span proxy).
- **Abstention TNR** — on unanswerable questions, how often we correctly decline.
- **Selective Accuracy / Coverage** — accuracy and volume at a confidence threshold.
- **ECE** — calibration; 0 is perfect (confidence ≈ accuracy).
""")
