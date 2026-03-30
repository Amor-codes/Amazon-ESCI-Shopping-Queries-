# E-Commerce Search Relevancy with Amazon ESCI

A progressive, end-to-end search relevancy pipeline built on the **Amazon ESCI Shopping Queries Dataset**. The project walks through four generations of retrieval and ranking systems — from a BM25 baseline all the way to cross-encoder reranking — measuring each improvement rigorously with NDCG, MRR, and Precision.

---

## Overview

| Stage | System | Key Technique |
|-------|--------|---------------|
| 1 | BM25 Baseline | Okapi BM25, tokenisation, field boosting |
| 2 | Semantic Search | Dense vectors via `all-MiniLM-L6-v2` + FAISS HNSW |
| 3 | Hybrid Search | BM25 + Semantic fused with Reciprocal Rank Fusion (RRF) |
| 4 | Learning to Rank | LambdaMART (`lightgbm` lambdarank) over engineered features |
| 5 | Cross-Encoder Rerank | `ms-marco-MiniLM-L-6-v2` as a final precision pass |

---

## Dataset

**Amazon ESCI (E-Commerce Shopping Queries Dataset)** — published by Amazon Research (2022).

- **130,000+** real Amazon shopping queries across English, Spanish, and Japanese
- **1.8 million** (query, product) pairs with human-annotated relevance labels
- **Graded relevance** using four labels:

| Label | Grade | Meaning |
|-------|-------|---------|
| E — Exact | 3 | Precisely what the user wants |
| S — Substitute | 2 | Different but could satisfy the need |
| C — Complement | 1 | Goes well with the queried item |
| I — Irrelevant | 0 | Off-topic |

The notebook works with English (`us` locale) and samples 2,000 queries for Colab compatibility.

**Sources:** [Amazon Research GitHub](https://github.com/amazon-science/esci-data) · [HuggingFace `tasksource/esci`](https://huggingface.co/datasets/tasksource/esci)

---

## Pipeline Architecture

```
Query
  │
  ├─► BM25 Retrieval (top-50 candidates)
  │       └─ Standard tokeniser: lowercase → strip punctuation → split
  │
  ├─► Semantic Retrieval (top-50 candidates)
  │       └─ all-MiniLM-L6-v2 (384-dim) → FAISS HNSW index
  │
  ├─► RRF Fusion (Hybrid)
  │       └─ score(d) = Σ 1/(k + rank_i),  k = 60
  │
  ├─► LTR Reranking (top-50 → reranked)
  │       └─ LambdaMART via LightGBM
  │           ├─ BM25 per-field scores (title, full text)
  │           ├─ Cosine similarity from embeddings
  │           ├─ Exact-match flags
  │           └─ Product quality signals
  │
  └─► Cross-Encoder Reranking (top-50 → final top-10)
          └─ ms-marco-MiniLM-L-6-v2
              Input: [query] [SEP] [brand + title + description snippet]
```

---

## Project Structure

```
amazon-esci-shopping-queries.ipynb   # Main notebook (all 22 cells)

# Artifacts written to disk at runtime
product_embeddings.npy               # (n_products, 384) float32 array
products.parquet                     # Product metadata DataFrame
hnsw_product.index                   # Serialised FAISS HNSW index
feature_importance.png               # LTR feature importance chart
system_comparison.png                # Bar chart: all systems vs metrics
```

---

## Dependencies

Install everything with the first notebook cell, or run manually:

```bash
pip install rank_bm25==0.2.2 \
            sentence-transformers==2.7.0 \
            faiss-cpu==1.8.0 \
            lightgbm==4.3.0 \
            ranx==0.3.16 \
            datasets==2.19.0 \
            transformers==4.40.0 \
            symspellpy==6.7.7 \
            plotly==5.21.0 \
            seaborn matplotlib pandas numpy tqdm scikit-learn
```

**Runtime:** Google Colab (T4 GPU recommended) or Kaggle. CPU-only is supported but slower for the embedding and cross-encoder stages.

---

## Evaluation Metrics

All systems are evaluated using [ranx](https://github.com/AmenRa/ranx):

| Metric | What it measures |
|--------|-----------------|
| **NDCG@5** | Ranking quality in the top 5 results (graded relevance) |
| **NDCG@10** | Ranking quality in the top 10 results |
| **MRR@10** | How high the first relevant result appears |
| **Precision@3** | Fraction of top-3 results that are relevant |

The notebook also includes a per-query delta histogram (Cross-Encoder vs LTR) to visualise where the final reranking stage wins and where it degrades.

---

## Key Design Decisions

**Field boosting via repetition** — the product title is concatenated twice in `product_text` to replicate Elasticsearch field-weight boosting without requiring a custom analyser.

**BM25 parameters** — `k1=1.5, b=0.75` for full-text; `k1=1.2, b=0.3` for the title-only index (low `b` because titles are short).

**RRF constant** — `k=60`, matching the Elasticsearch 8.x default, making the hybrid scores directly comparable to production deployments.

**GroupShuffleSplit for LTR** — queries are split at the group level so no query appears in both train and validation, preventing leakage.

**Cascade efficiency** — each stage retrieves more candidates than it returns (Hybrid: 50+50 → 50; LTR: 50 → 50; Cross-Encoder: 50 → 10), balancing recall with inference cost.

---

## Extending the Project

- **Multilingual support** — swap `all-MiniLM-L6-v2` for `multilingual-e5-large` or `MuRIL` to handle Spanish and Japanese queries from the full ESCI dataset.
- **Full dataset** — remove the `QUERY_SAMPLE = 2000` cap and run on a machine with sufficient RAM (≥ 32 GB recommended for the full 1.8M pairs).
- **Elasticsearch integration** — the HNSW parameters (`M=16`, `efConstruction=200`, `efSearch=50`) map directly to ES 8.x `dense_vector` mapping fields; the RRF fusion maps to the ES `rrf` rank clause.
- **Additional LTR features** — query-category overlap, product review count, historical click-through rate, or spell-corrected query signals (SymSpellPy is already installed).

---

## References

- Reddy et al. (2022). *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search.* [arXiv:2206.06588](https://arxiv.org/abs/2206.06588)
- Burges (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research.
- Johnson et al. (2021). *Billion-Scale Similarity Search with GPUs.* IEEE Transactions on Big Data — [FAISS](https://github.com/facebookresearch/faiss).
- Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
