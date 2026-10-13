# RAG Retrieval Lab

A small, runnable benchmark of the **retrieval** half of RAG. It compares four retrievers on a labeled question set, then optionally answers with Claude from the top passages. No GPU, no model downloads and no API key needed for the benchmark.

```
python rag.py eval

retriever   hit@1  hit@3    MRR   (20 questions, 16 passages)
tfidf        0.90   0.95   0.93
bm25         0.90   0.95   0.93
lsa          0.85   1.00   0.93
hybrid       0.85   0.95   0.91
```

## What's in it

| Retriever | Idea |
|---|---|
| `tfidf` | Cosine similarity over TF-IDF vectors (scikit-learn) |
| `bm25` | Okapi BM25, implemented from scratch in about 15 lines |
| `lsa` | TF-IDF compressed with SVD, a tiny stand-in for dense embeddings |
| `hybrid` | Reciprocal rank fusion of the three rankings |

Metrics are **hit@1**, **hit@3** and **MRR** (mean reciprocal rank) against `data/questions.jsonl`. The questions are deliberately paraphrased, so keyword overlap with the passages is low.

## Reading the results

On this small corpus the retrievers are close, and the hybrid doesn't win. That is expected with 16 passages and 20 questions: one question moving changes a score by 0.05, so the differences are within noise. The point of the lab is the harness. Add your own passages and questions to `data/` and the table shows what works for your data. LSA does reach perfect hit@3, which hints at the value of semantic matching on paraphrased queries.

## Answer generation (optional)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python rag.py ask "why does my loss blow up during training?"
```

The hybrid retriever picks the top 3 passages, and Claude answers from them with `[p02]`-style citations.

## Layout

```
rag.py                  retrievers, evaluation and generation (one file)
data/corpus.jsonl       16 short ML passages
data/questions.jsonl    20 labeled questions
```

## Ideas to extend

- Swap `LSA` for real embeddings and compare the numbers.
- Add a chunking-size sweep.
- Add an LLM judge to score answer faithfulness, not just retrieval.
