#!/usr/bin/env python3

import json
import pandas as pd
import faiss
import os
from metrics import evaluate_metrics
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# --- Step 1: Log in to Hugging Face ---
# You will be prompted to paste your Hugging Face token
login(token="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")

# --- Step 2: Mount Google Drive & load JSON ---
from google.colab import drive
drive.mount('/content/drive')

json_path = "/content/drive/MyDrive/CS792 Research Thesis/Data/social_security_act.json"

with open(json_path, "r") as f:
    data = json.load(f)

# --- Step 3: Flatten JSON into DataFrame ---
df = pd.DataFrame([
    {"section": elem["id"], "text": elem["c"], "url": elem["url"]}
    for elem in data if elem["c"].strip() != ""
])

# --- Step 4: Chunk text ---
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i + chunk_size])

chunks = []
for _, row in df.iterrows():
    for chunk in chunk_text(row["text"]):
        chunks.append({"section": row["section"], "text": chunk, "url": row["url"]})

chunks_df = pd.DataFrame(chunks)
print("Chunks DataFrame sample:")
print(chunks_df.head())

# --- Step 5: Build FAISS index ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks_df["text"].tolist(), convert_to_numpy=True, batch_size=64)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

def retrieve(query, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return chunks_df.iloc[I[0]]

# --- Step 6: Load LLaMA 3.2 - 3B Instruct with HF token ---
model_id = "meta-llama/Llama-3.2-3b-instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    token=True
)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.1
)

def ask_llama(prompt):
    out = llama_pipeline(prompt, return_full_text=True)
    return out[0]["generated_text"]

# --- Step 7: Compare Baseline vs RAG ---
query = "Am I eligible for Jobseeker Support if I can't work due to health reasons?"

# Baseline (no retrieval)
baseline_answer = ask_llama(query)
print("=== BASELINE ===\n", baseline_answer, "\n")

# RAG (retrieval + citations)
retrieved = retrieve(query, k=3)
context = "\n".join([f"[{r.section}: {r.text}]" for _, r in retrieved.iterrows()])
rag_prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}"

rag_answer = ask_llama(rag_prompt)
print("=== RAG ===\n", rag_answer)
