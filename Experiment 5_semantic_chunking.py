#!/usr/bin/env python3
import json
import os
import re
import faiss
import numpy as np
import pandas as pd

from bm25lite import BM25Lite
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# 🔹 NEW: import metrics helper
from metrics import compute_all_metrics

# ----------------------------------
# Hugging Face login (env variable)
# ----------------------------------
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("WARNING: HUGGINGFACE_HUB_TOKEN not set.")

# ----------------------------------
# Load SSA JSON
# ----------------------------------
JSON_PATH = "social_security_act.json"
with open(JSON_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame(
    [
        {"id": d["id"], "c": d["c"], "url": d.get("url", "")}
        for d in data
        if d["c"].strip() != ""
    ]
)

# ----------------------------------
# Metadata parser
# ----------------------------------
def parse_id(v: str):
    m = {"part": None, "subpart": None, "section": None, "subsection": None}
    p = re.search(r"part(\d+)", v, re.I)
    s = re.search(r"subpart(\d+)", v, re.I)
    sec = re.search(r"s(\d+)", v, re.I)
    ln = re.search(r"line(\d+)", v, re.I)

    if p:
        m["part"] = p.group(1)
    if s:
        m["subpart"] = s.group(1)
    if sec:
        m["section"] = sec.group(1)
    if ln:
        m["subsection"] = ln.group(1)

    return m

# ----------------------------------
# Simple sentence splitter
# ----------------------------------
def sent_tokenize_simple(text: str):
    # crude but robust enough for statute prose
    sents = re.split(r"(?<=[\.\?\!;:])\s+", text)
    return [s.strip() for s in sents if s.strip()]

# ----------------------------------
# Rhetorical-role keywords
# ----------------------------------
KEYWORDS_DEFINITION = [
    "in this section",
    "for the purposes of this",
    "for the purposes of this act",
    "for the purposes of",
    "means",
    "includes",
]

KEYWORDS_EXCEPTION = [
    "despite",
    "does not apply",
    "except that",
    "unless",
    "however",
]

KEYWORDS_ENTITLEMENT = [
    "is entitled to",
    "is eligible for",
    "may be granted",
    "may receive",
    "qualifies for",
]

KEYWORDS_OBLIGATION = [
    "must",
    "shall",
    "is required to",
    "is obliged to",
]

def is_boundary_sentence(sent: str) -> bool:
    """Heuristic: start of a new rhetorical unit."""
    s = sent.strip()
    low = s.lower()

    # numbered subsection markers like "(1)", "(2A)", "1.", "2."
    if re.match(r"^\(?\d+[A-Za-z]?\)", s):
        return True

    # definition / exception boundaries are strong cut points
    if any(kw in low for kw in KEYWORDS_DEFINITION + KEYWORDS_EXCEPTION):
        return True

    return False

def classify_role(sent: str) -> str:
    low = sent.lower()
    if any(kw in low for kw in KEYWORDS_DEFINITION):
        return "definition"
    if any(kw in low for kw in KEYWORDS_EXCEPTION):
        return "exception"
    if any(kw in low for kw in KEYWORDS_ENTITLEMENT):
        return "entitlement"
    if any(kw in low for kw in KEYWORDS_OBLIGATION):
        return "obligation"
    return "other"

# ----------------------------------
# ChuLo-Lite v2: section-aware semantic chunking
# ----------------------------------
def chulo_lite_v2(text: str, max_tokens: int = 220, min_tokens: int = 40):
    """
    Section-aware, rhetorically-informed chunker:
    - Splits into sentences
    - Starts new chunk on rhetorical boundaries
    - Enforces max_tokens and avoids tiny chunks when possible
    """
    sentences = sent_tokenize_simple(text)
    chunks = []
    current = []

    def cur_len():
        return sum(len(s.split()) for s in current)

    for sent in sentences:
        # if this sentence is a boundary and current chunk is reasonably sized, flush
        if is_boundary_sentence(sent) and cur_len() >= min_tokens:
            chunks.append(" ".join(current))
            current = []

        current.append(sent)

        # hard length cap
        if cur_len() >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    # simple post-pass: merge trailing tiny chunks into previous if needed
    merged = []
    for ch in chunks:
        if not merged:
            merged.append(ch)
            continue
        if len(ch.split()) < min_tokens:
            merged[-1] = merged[-1] + " " + ch
        else:
            merged.append(ch)

    return merged

# ----------------------------------
# Build section-level text map
# ----------------------------------
sections_map = {}

for _, row in df.iterrows():
    labels = parse_id(row["id"])
    key = (labels["part"], labels["subpart"], labels["section"])

    if key not in sections_map:
        sections_map[key] = {
            "part": labels["part"],
            "subpart": labels["subpart"],
            "section": labels["section"],
            "id": row["id"],   # first id as representative
            "url": row["url"],
            "texts": [],
        }

    sections_map[key]["texts"].append(row["c"])

# ----------------------------------
# Build semantic chunks DF (ChuLo-Lite v2)
# ----------------------------------
chunk_rows = []

for key, rec in sections_map.items():
    part = rec["part"]
    subpart = rec["subpart"]
    section = rec["section"]
    rep_id = rec["id"]
    url = rec["url"]

    full_text = " ".join(rec["texts"]).strip()
    if not full_text:
        continue

    # For rows that don't belong to a specific section (e.g. preamble),
    # just keep the whole text as one chunk so we don't mangle it.
    if section is None:
        chunk_texts = [full_text]
    else:
        chunk_texts = chulo_lite_v2(full_text, max_tokens=220, min_tokens=40)

    # citation label at section level
    citation = "SSA 2018"
    if part:
        citation += f" – Part {part}"
    if subpart:
        citation += f", Subpart {subpart}"
    if section:
        citation += f", Section {section}"

    for ch in chunk_texts:
        ch = ch.strip()
        if not ch:
            continue

        sents = sent_tokenize_simple(ch)
        first_sent = sents[0] if sents else ""
        role = classify_role(first_sent) if first_sent else "other"

        chunk_rows.append(
            {
                "id": rep_id,
                "part": part,
                "subpart": subpart,
                "section": section,
                "subsection": None,  # not used at this level
                "citation_label": citation,
                "role": role,
                "text": ch,
                "url": url,
            }
        )

chunks_df = pd.DataFrame(chunk_rows)

# ----------------------------------
# Embeddings (cosine; MiniLM)
# ----------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(chunks_df["text"].tolist(), convert_to_numpy=True)

faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# ----------------------------------
# BM25Lite (lexical)
# ----------------------------------
tokenized = [t.lower().split() for t in chunks_df["text"].tolist()]
bm25 = BM25Lite(tokenized)

# ----------------------------------
# Hybrid retrieval
# ----------------------------------
def hybrid_retrieve(query: str, k: int = 3, alpha: float = 0.65):
    """
    alpha ~ weight for dense (MiniLM); (1-alpha) for lexical (BM25Lite).
    Returns top-k semantic chunks.
    """
    qemb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qemb)
    D, I = index.search(qemb, len(chunks_df))
    dense = D[0]

    # normalise dense scores
    if dense.max() > dense.min():
        dense = (dense - dense.min()) / (dense.max() - dense.min())
    else:
        dense = np.zeros_like(dense)

    # lexical scores
    lex = np.array(bm25.get_scores(query.lower().split()))
    if lex.max() > lex.min():
        lex = (lex - lex.min()) / (lex.max() - lex.min())
    else:
        lex = np.zeros_like(lex)

    combined = alpha * dense + (1 - alpha) * lex
    topk = np.argsort(combined)[-k:][::-1]
    return chunks_df.iloc[topk]

# ----------------------------------
# Safe truncation
# ----------------------------------
def truncate_words(text: str, max_words: int):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

# ----------------------------------
# LLaMA generator
# ----------------------------------
model_id = "meta-llama/Llama-3.2-3b-instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

llama = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.1,
)

def ask(prompt: str) -> str:
    return llama(prompt, return_full_text=True)[0]["generated_text"]

# ----------------------------------
# MAIN (Experiment 5 – ChuLo-Lite v2)
# ----------------------------------
if __name__ == "__main__":
    query = "Am I eligible for Jobseeker Support if I can't work due to health reasons?"

    retrieved = hybrid_retrieve(query, k=3)
    sections = retrieved["section"].dropna().unique().tolist()

    merged_sections = []

    for sec in sections:
        # reconstruct full section text from original df
        sec_rows = df[df["id"].str.contains(f"s{sec}", case=False, regex=True)]
        if sec_rows.empty:
            continue

        first_id = sec_rows.iloc[0]["id"]
        lbl = parse_id(first_id)
        part, subpart = lbl["part"], lbl["subpart"]

        citation = "SSA 2018"
        if part:
            citation += f" – Part {part}"
        if subpart:
            citation += f", Subpart {subpart}"
        citation += f", Section {sec}"

        full_text = " ".join(sec_rows["c"].tolist())
        full_text = truncate_words(full_text, max_words=2000)  # per-section safety

        merged_sections.append(f"[{citation}]: {full_text}")

    # final context cap
    context = "\n\n".join(merged_sections)
    context = truncate_words(context, max_words=6000)

    rag_prompt = f"""
You are an assistant specialised in New Zealand social security law.
Answer STRICTLY from the provided context (SSA 2018).
Do NOT reference UK law, overseas systems, or repealed benefits.

Context:
{context}

Question:
{query}

Instructions:
1) Think step-by-step using ONLY the context.
2) Produce a single-sentence final answer.
3) Provide 2–4 sentences of reasoning.
4) Cite specific SSA 2018 sections used.

Output format:
Answer: ...
Reasoning: ...
Citations: [...]
"""

    print("\n=== RAG (Experiment 5 — ChuLo-Lite v2 + Hybrid Retrieval) ===\n")
    # 🔹 capture answer instead of printing inline
    model_answer = ask(rag_prompt)
    print(model_answer)

    # 🔹 METRICS HOOK (minimal)
    # manually defined gold signal for this one query
    gold_sections = ["21", "27"]  # adjust if you refine your gold
    ref_answer = (
        "You may be eligible for Jobseeker Support if your health condition "
        "affects your ability to work and you meet Section 21 or 27 requirements."
    )
    retrieved_sections = [str(s) for s in sections]

    metrics = compute_all_metrics(
        pred_answer=model_answer,
        ref_answer=ref_answer,
        retrieved_sections=retrieved_sections,
        gold_sections=gold_sections,
        df=df,
        section_number=gold_sections[0],
        retrieved_rows=retrieved,
        k=3,
    )

    print("\n=== METRICS ===")
    for name, val in metrics.items():
        # handle non-float metrics gracefully
        try:
            print(f"{name}: {float(val):.4f}")
        except (TypeError, ValueError):
            print(f"{name}: {val}")
