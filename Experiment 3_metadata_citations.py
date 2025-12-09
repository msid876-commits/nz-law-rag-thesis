#!/usr/bin/env python3

import json
import pandas as pd
import re
import faiss
import os
from metrics import evaluate_metrics
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# HF LOGIN
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)

JSON_PATH = "social_security_act.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame([
    {"id": elem["id"], "c": elem["c"], "url": elem.get("url", "")}
    for elem in data if elem["c"].strip() != ""
])

def parse_id(idv):
    m = {
        "part": None, "subpart": None,
        "section": None, "subsection": None
    }
    p = re.search(r"part(\d+)", idv, re.I)
    s = re.search(r"subpart(\d+)", idv, re.I)
    sec = re.search(r"s(\d+)", idv, re.I)
    ln = re.search(r"line(\d+)", idv, re.I)

    if p: m["part"] = p.group(1)
    if s: m["subpart"] = s.group(1)
    if sec: m["section"] = sec.group(1)
    if ln: m["subsection"] = ln.group(1)
    return m

def chunk_text(t, size=300, ov=50):
    w = t.split()
    for i in range(0, len(w), size-ov):
        yield " ".join(w[i:i+size])

chunks = []
for _, row in df.iterrows():
    labels = parse_id(row["id"])
    citation = "SSA 2018"
    if labels["part"]: citation += f" – Part {labels['part']}"
    if labels["subpart"]: citation += f", Subpart {labels['subpart']}"
    if labels["section"]: citation += f", Section {labels['section']}"
    if labels["subsection"]: citation += f", Line {labels['subsection']}"

    for chunk in chunk_text(row["c"]):
        chunks.append({
            "chunk_id": row["id"],
            "part": labels["part"],
            "subpart": labels["subpart"],
            "section": labels["section"],
            "subsection": labels["subsection"],
            "citation_label": citation,
            "text": chunk,
            "url": row["url"]
        })

chunks_df = pd.DataFrame(chunks)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(chunks_df["text"].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

def retrieve(q, k=10):
    q_emb = embedder.encode([q], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return chunks_df.iloc[I[0]]

model_id = "meta-llama/Llama-3.2-3b-instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
llama = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512)

def ask(p):
    return llama(p, return_full_text=True)[0]["generated_text"]

if __name__ == "__main__":
    query = "Am I eligible for Jobseeker Support if I can't work due to health reasons?"
    retrieved = retrieve(query)

    context = "\n".join([
        f"[{r.citation_label}]: {r.text}"
        for _, r in retrieved.iterrows()
    ])    
    rag_prompt = f"Use this context:\n{context}\n\nQuestion: {query}"
    print("=== RAG ===")
    rag_answer = ask(rag_prompt)
    print(rag_answer)
    results = evaluate_metrics(rag_answer)
    print("\n=== METRICS ===")
    print(results)