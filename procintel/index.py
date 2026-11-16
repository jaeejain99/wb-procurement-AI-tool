"""Hybrid retrieval index: sparse TF-IDF + dense LSA embeddings, metadata pre-filters, RRF fusion, soft boosts.
The Embedder is deliberately offline/dependency-light; swap `_fit_dense` for a sentence-transformer / API embedder in production."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def soft_boost(c, prefer, now_year=2026, feedback=None) -> float:
    """Metadata/recency/feedback prior shared by every retrieval backend (local index and Databricks Vector Search)."""
    b = 1.0
    b *= 1.25 if prefer.get("sector") and c.sector == prefer["sector"] else 1
    b *= 1.15 if prefer.get("country") and c.country == prefer["country"] else 1
    b *= 1.20 if prefer.get("section_kind") and c.section_kind == prefer["section_kind"] else 1
    try:
        b *= 1 - 0.01 * min(15, max(0, now_year - int(c.date[:4])))     # mild recency decay
    except ValueError:
        pass
    return b * (1 + (feedback or {}).get(c.chunk_id, 0.0))


class HybridIndex:
    def __init__(self, dims=96, oov_refit=0.20, growth_refit=0.25):
        self.dims, self.oov_refit, self.growth_refit = dims, oov_refit, growth_refit
        self.chunks: list = []
        self.boost: dict[str, float] = {}      # feedback-driven priors
        self._base_n = 0

    # -- build
    def _fit(self):
        texts = [c.embed_text for c in self.chunks]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.X = self.vec.fit_transform(texts)
        k = max(2, min(self.dims, self.X.shape[1] - 1, self.X.shape[0] - 1))
        self.svd = TruncatedSVD(k, random_state=0)
        self.Z = normalize(self.svd.fit_transform(self.X))
        self._base_n = len(self.chunks)

    def build(self, chunks):
        self.chunks = list(chunks)
        self.ids = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._fit()
        return self

    def add(self, new_chunks) -> dict:
        """Incremental, idempotent upsert (content-hash dedupe). Cheap transform with frozen vocabulary; triggers a full
        refit when the out-of-vocabulary rate or index growth says the embedding space is stale."""
        fresh = [c for c in new_chunks if c.chunk_id not in self.ids or self.chunks[self.ids[c.chunk_id]].content_hash != c.content_hash]
        if not fresh:
            return dict(added=0, refit=False, oov=0.0)
        analyzer = self.vec.build_analyzer()
        toks = [t for c in fresh for t in analyzer(c.embed_text)]
        vocab = self.vec.vocabulary_
        oov = float(np.mean([t not in vocab for t in toks])) if toks else 0.0
        for c in fresh:
            if c.chunk_id in self.ids:                 # updated doc version: replace in place
                self.chunks[self.ids[c.chunk_id]] = c
            else:
                self.ids[c.chunk_id] = len(self.chunks); self.chunks.append(c)
        refit = oov > self.oov_refit or (len(self.chunks) - self._base_n) / max(self._base_n, 1) > self.growth_refit
        if refit:
            self._fit()
        else:
            self.X = self.vec.transform([c.embed_text for c in self.chunks])
            self.Z = normalize(self.svd.transform(self.X))
        return dict(added=len(fresh), refit=refit, oov=round(oov, 3))

    # -- search
    def search(self, query, k=8, filters=None, prefer=None, now_year=2026):
        """filters: hard metadata constraints, e.g. {'doc_type': {'ICR','PPAR'}}.
        prefer: soft boosts, e.g. {'sector': 'Transport', 'country': 'Ghana', 'section_kind': 'lessons'}."""
        filters, prefer = filters or {}, prefer or {}
        mask = np.array([all(getattr(c, f) in v for f, v in filters.items()) for c in self.chunks])
        if not mask.any():
            return []
        qx = self.vec.transform([query])
        sparse = (self.X @ qx.T).toarray().ravel()
        dense = self.Z @ normalize(self.svd.transform(qx)).ravel()
        idx = np.where(mask)[0]
        rr = lambda s: {i: r for r, i in enumerate(idx[np.argsort(-s[idx])])}
        rs, rd = rr(sparse), rr(dense)
        mu, sd = sparse.mean(), sparse.std() or 1.0     # relevance vs the WHOLE corpus (scale-free across corpus sizes)
        out = []
        for i in idx:
            c = self.chunks[i]
            rrf = 1 / (60 + rs[i]) + 1 / (60 + rd[i])
            boost = soft_boost(c, prefer, now_year, self.boost)
            out.append((c, rrf * boost * 100, dict(sparse=float(sparse[i]), dense=float(dense[i]), sparse_z=float((sparse[i] - mu) / sd))))
        out.sort(key=lambda t: -t[1])
        return out[:k]
