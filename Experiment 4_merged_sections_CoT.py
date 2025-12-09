#!/usr/bin/env python3

import json
import pandas as pd
import faiss
import re
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# 🔹 NEW: metrics import
from metrics import compute_all_metrics

# -----------------------------
# Hugging Face login (from env)
# -----------------------------
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("WARNING: HUGGINGFACE_HUB_TOKEN not set in environment.")

# -----------------------------
# Load SSA JSON
# -----------------------------
JSON_PATH = "social_security_act.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame([
    {"id": elem["id"], "c": elem["c"], "url": elem.get("url", "")}
    for elem in data if elem["c"].strip() != ""
])

# -----------------------------
# Metadata parser (full: part, subpart, section, line)
# -----------------------------
def parse_id(idv: str):
    m = {"part": None, "subpart": None, "section": None, "subsection": None}
    p = re.search(r"part(\d+)", idv, re.I)
    s = re.search(r"subpart(\d+)", idv, re.I)
    sec = re.search(r"s(\d+)", idv, re.I)
    ln = re.search(r"line(\d+)", idv, re.I)

    if p:
        m["part"] = p.group(1)
    if s:
        m["subpart"] = s.group(1)
    if sec:
        m["section"] = sec.group(1)
    if ln:
        m["subsection"] = ln.group(1)

    return m

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(t: str, size: int = 300, ov: int = 50):
    w = t.split()
    step = max(size - ov, 1)
    for i in range(0, len(w), step):
        yield " ".join(w[i:i+size])

# -----------------------------
# Build chunks with metadata + citation_label
# -----------------------------
chunks = []
for _, row in df.iterrows():
    labels = parse_id(row["id"])
    part = labels["part"]
    subpart = labels["subpart"]
    section = labels["section"]
    subsection = labels["subsection"]

    # citation label like: SSA 2018 – Part 3, Subpart 2, Section 44, Line 3
    citation = "SSA 2018"
    if part:
        citation += f" – Part {part}"
    if subpart:
        citation += f", Subpart {subpart}"
    if section:
        citation += f", Section {section}"
    if subsection:
        citation += f", Line {subsection}"

    for ch in chunk_text(row["c"]):
        chunks.append({
            "id": row["id"],
            "part": part,
            "subpart": subpart,
            "section": section,
            "subsection": subsection,
            "citation_label": citation,
            "text": ch,
            "url": row["url"]
        })

chunks_df = pd.DataFrame(chunks)

# -----------------------------
# FAISS index on chunk text
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(
    chunks_df["text"].tolist(),
    convert_to_numpy=True
)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

def retrieve(q: str, k: int = 3):
    qemb = embedder.encode([q], convert_to_numpy=True)
    D, I = index.search(qemb, k)
    return chunks_df.iloc[I[0]]

# -----------------------------
# LLaMA model
# -----------------------------
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

def ask(p: str) -> str:
    return llama(p, return_full_text=True)[0]["generated_text"]

# -----------------------------
# Main: merged-section RAG with metadata + CoT prompting
# -----------------------------
if __name__ == "__main__":
    query = "Am I eligible for Jobseeker Support if I can't work due to health reasons?"

    # retrieve top-k chunks
    ret = retrieve(query, k=3)

    # which section numbers were retrieved?
    sections = ret["section"].dropna().unique().tolist()

    # build merged, metadata-aware context
    merged_context_lines = []

    for sec in sections:
        rows = df[df["id"].str.contains(f"s{sec}", case=False, regex=True)]
        if rows.empty:
            continue

        raw_id = rows.iloc[0]["id"]
        labels = parse_id(raw_id)
        part = labels["part"]
        subpart = labels["subpart"]

        # citation label at merged-section level
        cl = "SSA 2018"
        if part:
            cl += f" – Part {part}"
        if subpart:
            cl += f", Subpart {subpart}"
        cl += f", Section {sec}"

        full_text = " ".join(rows["c"].tolist())
        merged_context_lines.append(f"[{cl}]: {full_text}")

    context = "\n".join(merged_context_lines)

    # -----------------------------
    # RAG (Merged Sections) with CoT-style prompt
    # -----------------------------
    rag_prompt = f"""
You are an assistant specialized in New Zealand social security law.
Answer STRICTLY from the provided context (excerpts from the NZ Social Security Act 2018).
Do NOT reference UK benefits (JSA/UC) or anything not in context.
If the context is insufficient, say so and name the missing section(s).

Context:
{context}

Question:
{query}

Instructions:
1) Think through the rules in the context and apply them to the question.
2) Then produce a concise final answer and cite the specific SSA 2018 section numbers used.

Output format (exact labels):
Answer: <one-sentence conclusion>
Reasoning (brief): <2–4 sentences applying the cited rules to the question>
Citations: [SSA 2018 – Part X, Subpart Y, Section Z; ...]
"""

    rag_answer = ask(rag_prompt)
    print("=== RAG (Merged Sections, CoT Prompt) ===\n")
    print(rag_answer)

    # -----------------------------
    # METRICS (minimal hook)
    # -----------------------------
    # manually specify what “gold” looks like for this question
    gold_sections = ["21", "27"]   # adjust if you settle on slightly different gold
    ref_answer = (
        "You may be eligible for Jobseeker Support if your health condition "
        "affects your ability to work and you meet Section 21 or 27 requirements."
    )
    retrieved_sections = [str(s) for s in sections]

    metrics = compute_all_metrics(
        pred_answer=rag_answer,
        ref_answer=ref_answer,
        retrieved_sections=retrieved_sections,
        gold_sections=gold_sections,
        df=df,
        section_number=gold_sections[0],  # e.g. "21"
        retrieved_rows=ret,
        k=3
    )

    print("\n=== METRICS (Experiment 4) ===")
    for name, val in metrics.items():
        try:
            print(f"{name}: {float(val):.4f}")
        except (TypeError, ValueError):
            print(f"{name}: {val}")
