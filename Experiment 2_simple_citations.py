#!/usr/bin/env python3

import json
import pandas as pd
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
    {"section": elem["id"], "text": elem["c"], "url": elem.get("url", "")}
    for elem in data if elem["c"].strip() != ""
])

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i + chunk_size])

chunks = []
for _, row in df.iterrows():
    citation = f"SSA 2018 – Section {row['section'].upper()}"
    for chunk in chunk_text(row["text"]):
        chunks.append({
            "section": row["section"],
            "citation": citation,
            "text": chunk,
            "url": row["url"]
        })

chunks_df = pd.DataFrame(chunks)

# FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(chunks_df["text"].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

def retrieve(q, k=3):
    q_emb = embedder.encode([q], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return chunks_df.iloc[I[0]]

# LLM
model_id = "meta-llama/Llama-3.2-3b-instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
llama = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512)

def ask(p):
    return llama(p, return_full_text=True)[0]["generated_text"]

# RUN
if __name__ == "__main__":
    query = "Am I eligible for Jobseeker Support if I can't work due to health reasons?"
    retrieved = retrieve(query)

    context = "\n".join([f"[{r.citation}]: {r.text}" for _, r in retrieved.iterrows()])
    rag_prompt = f"Use this context:\n{context}\n\nQuestion: {query}"
    print("=== RAG ===")
    rag_answer = ask(rag_prompt)
    print(rag_answer)
    results = evaluate_metrics(rag_answer)
    print("\n=== METRICS ===")
    print(results)