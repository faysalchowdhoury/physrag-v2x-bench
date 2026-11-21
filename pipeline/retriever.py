"""Physics-conditioned hybrid retriever over the SOP knowledge base.

Combines BM25 (lexical) and dense TF-IDF cosine (semantic). When the physics
gate supplies regime tags, matching SOPs are boosted -- so a spoofed alert whose
*text* looks benign is still routed to the correct high-severity SOPs via the
verified physical regime.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> List[str]:
    return _TOKEN.findall(s.lower())


class HybridRetriever:
    def __init__(self, kb_path: Optional[str] = None, regime_boost: float = 0.6):
        if kb_path is None:
            kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "kb", "v2x_sop_corpus.json")
        with open(kb_path, "r", encoding="utf-8") as fh:
            self.kb = json.load(fh)
        self.ids = [s["sop_id"] for s in self.kb]
        self.by_id = {s["sop_id"]: s for s in self.kb}
        self.docs = [f"{s['title']} {s['text']} {' '.join(s['keywords'])}" for s in self.kb]
        self.regime_boost = regime_boost
        # dense TF-IDF
        self._tfidf = TfidfVectorizer().fit(self.docs)
        self._doc_vecs = self._tfidf.transform(self.docs)
        # BM25 statistics
        self._tokens = [_tok(d) for d in self.docs]
        self._len = np.array([len(t) for t in self._tokens], dtype=float)
        self._avgdl = self._len.mean()
        self._tf = [Counter(t) for t in self._tokens]
        df = Counter()
        for t in self._tokens:
            df.update(set(t))
        self._idf = {w: math.log(1 + (len(self.docs) - c + 0.5) / (c + 0.5))
                     for w, c in df.items()}

    def _bm25(self, q_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(len(self.docs))
        for i, tf in enumerate(self._tf):
            s = 0.0
            for w in q_tokens:
                if w in tf:
                    idf = self._idf.get(w, 0.0)
                    num = tf[w] * (k1 + 1)
                    den = tf[w] + k1 * (1 - b + b * self._len[i] / self._avgdl)
                    s += idf * num / den
            scores[i] = s
        return scores

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    def retrieve(self, query: str, regime_tags: Optional[List[str]] = None, k: int = 3
                 ) -> List[str]:
        bm = self._norm(self._bm25(_tok(query)))
        dense = self._norm(cosine_similarity(self._tfidf.transform([query]),
                                             self._doc_vecs).ravel())
        score = 0.5 * bm + 0.5 * dense
        if regime_tags:                                   # physics-conditioned boost
            boost = np.array([self.regime_boost if any(r in self.by_id[i]["regime_tags"]
                                                       for r in regime_tags) else 0.0
                              for i in self.ids])
            score = score + boost
        order = np.argsort(-score)[:k]
        return [self.ids[i] for i in order]
