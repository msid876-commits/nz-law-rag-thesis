# metrics.py
"""
Evaluation utilities for NZ Social Law RAG:
- ROUGE-L (sequence-based overlap)
- BERTScore (semantic similarity)
- Chunk-level retrieval metrics:
    * Precision@K
    * Recall@K
    * Section Coverage Score
"""

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bertscore


# ------------------------------------------------------
# 1. ROUGE-L Score (recommended for legal text)
# ------------------------------------------------------
def compute_rouge_l(pred: str, ref: str):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return scores["rougeL"].fmeasure


# ------------------------------------------------------
# 2. BERTScore (semantic similarity)
# ------------------------------------------------------
def compute_bertscore(pred: str, ref: str, lang="en"):
    P, R, F1 = bertscore(
        [pred],
        [ref],
        lang=lang,
        rescale_with_baseline=True
    )
    return float(F1[0])


# ------------------------------------------------------
# 3. Precision@K
# ------------------------------------------------------
def precision_at_k(retrieved_sections, gold_sections, k):
    """
    retrieved_sections : list of section numbers returned by retrieval
    gold_sections : list of section numbers that are truly relevant
    """
    retrieved_set = set(retrieved_sections[:k])
    gold_set = set(gold_sections)

    if k == 0:
        return 0.0

    return len(retrieved_set & gold_set) / k


# ------------------------------------------------------
# 4. Recall@K
# ------------------------------------------------------
def recall_at_k(retrieved_sections, gold_sections, k):
    retrieved_set = set(retrieved_sections[:k])
    gold_set = set(gold_sections)

    if len(gold_set) == 0:
        return 0.0

    return len(retrieved_set & gold_set) / len(gold_set)


# ------------------------------------------------------
# 5. Section Coverage Score
# ------------------------------------------------------
def section_coverage_score(df, section_number, retrieved_rows):
    """
    df: pandas DataFrame (full Act)
    section_number: e.g. "141"
    retrieved_rows: rows from df or chunks_df returned by retrieval
    """

    # All subsections in Act
    all_rows = df[df["id"].str.contains(f"s{section_number}", case=False)]
    all_subs = set(all_rows["id"].tolist())

    # Retrieved subsections
    retrieved_subs = set(retrieved_rows["id"].tolist())

    if len(all_subs) == 0:
        return 0.0

    return len(all_subs & retrieved_subs) / len(all_subs)


# ------------------------------------------------------
# 6. Wrapper – compute all metrics at once
# ------------------------------------------------------
def compute_all_metrics(pred_answer, ref_answer,
                        retrieved_sections, gold_sections,
                        df, section_number, retrieved_rows, k=3):
    """
    Returns a dictionary of metrics.
    """

    results = {
        "rougeL": compute_rouge_l(pred_answer, ref_answer),
        "bertscore": compute_bertscore(pred_answer, ref_answer),
        "precision@k": precision_at_k(retrieved_sections, gold_sections, k),
        "recall@k": recall_at_k(retrieved_sections, gold_sections, k),
        "section_coverage": section_coverage_score(
            df, section_number, retrieved_rows
        ),
    }

    return results
