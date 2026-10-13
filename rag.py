"""RAG retrieval lab: compare TF-IDF, BM25, LSA and a hybrid retriever, then optionally answer with Claude.

Usage:
    python rag.py eval                      # benchmark the retrievers (no API key needed)
    python rag.py ask "your question"       # retrieve with the hybrid retriever, answer with Claude
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent / "data"
CORPUS = [json.loads(line) for line in (DATA / "corpus.jsonl").read_text().splitlines()]
QUESTIONS = [json.loads(line) for line in (DATA / "questions.jsonl").read_text().splitlines()]
TEXTS = [d["text"] for d in CORPUS]
IDS = [d["id"] for d in CORPUS]


# --- Retrievers: each maps a query to one relevance score per document -----------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfIdf:
    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform(TEXTS)

    def scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.vec.transform([query]), self.matrix).ravel()


class BM25:
    """Okapi BM25 written from scratch (about 15 lines)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in TEXTS]
        self.lengths = np.array([sum(c.values()) for c in self.docs])
        df = Counter(w for c in self.docs for w in c)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for w in set(tokenize(query)) & self.idf.keys():
            tf = np.array([c[w] for c in self.docs])
            norm = self.k1 * (1 - self.b + self.b * self.lengths / self.lengths.mean())
            out += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
        return out


class LSA:
    """Latent semantic analysis: TF-IDF compressed with SVD, a tiny stand-in for dense embeddings."""

    def __init__(self, dims: int = 12):
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dims, random_state=0)
        self.matrix = self.svd.fit_transform(self.vec.fit_transform(TEXTS))

    def scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.svd.transform(self.vec.transform([query])), self.matrix).ravel()


class Hybrid:
    """Reciprocal rank fusion: combine the rankings of several retrievers."""

    def __init__(self, retrievers, k: int = 60):
        self.retrievers, self.k = retrievers, k

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(TEXTS))
        for r in self.retrievers:
            ranks = (-r.scores(query)).argsort().argsort()  # rank 0 = best
            fused += 1 / (self.k + ranks + 1)
        return fused


def top_k(retriever, query: str, k: int = 3) -> list[int]:
    return list((-retriever.scores(query)).argsort()[:k])


# --- Evaluation: hit@k and mean reciprocal rank against labeled questions --------------------------
def evaluate(retriever) -> dict:
    hits1 = hits3 = rr = 0.0
    for item in QUESTIONS:
        ranked = [IDS[i] for i in (-retriever.scores(item["q"])).argsort()]
        rank = ranked.index(item["answer"]) + 1
        hits1 += rank == 1
        hits3 += rank <= 3
        rr += 1 / rank
    n = len(QUESTIONS)
    return {"hit@1": hits1 / n, "hit@3": hits3 / n, "MRR": rr / n}


def build_retrievers() -> dict:
    tfidf, bm25, lsa = TfIdf(), BM25(), LSA()
    return {"tfidf": tfidf, "bm25": bm25, "lsa": lsa, "hybrid": Hybrid([tfidf, bm25, lsa])}


# --- Generation: answer from the retrieved passages (optional, needs ANTHROPIC_API_KEY) ------------
def ask(question: str) -> str:
    import anthropic  # imported lazily so `eval` works without the SDK or a key

    hybrid = build_retrievers()["hybrid"]
    context = "\n".join(f"[{IDS[i]}] {TEXTS[i]}" for i in top_k(hybrid, question))
    response = anthropic.Anthropic().messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        system="Answer using only the context. Cite passage ids like [p05]. Say so if the context is insufficient.",
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if command == "eval":
        print(f"{'retriever':<10} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}   ({len(QUESTIONS)} questions, {len(TEXTS)} passages)")
        for name, retriever in build_retrievers().items():
            m = evaluate(retriever)
            print(f"{name:<10} {m['hit@1']:>6.2f} {m['hit@3']:>6.2f} {m['MRR']:>6.2f}")
    elif command == "ask":
        print(ask(" ".join(sys.argv[2:])))
    else:
        sys.exit(__doc__)
