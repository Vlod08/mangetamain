# src/mangetamain/core/clustering/recipes_text.py
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from mangetamain.core.dataset import RecipesDataset


def build_tfidf_kmeans(
    corpus: pd.Series,
    k: int = 6,
    maxf: int = 8000,
    random_state: int = 42,
) -> tuple[KMeans, TfidfVectorizer]:
    tfidf = TfidfVectorizer(max_features=maxf, stop_words="english")
    X = tfidf.fit_transform(corpus.astype(str))
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(X)
    return km, tfidf


def assign_clusters(df: pd.DataFrame, tfidf: TfidfVectorizer, km: KMeans, text_col: str = "text") -> pd.DataFrame:
    out = df.copy()
    if text_col not in out.columns:
        out[text_col] = ""
    X = tfidf.transform(out[text_col].astype(str))
    out["cluster"] = km.predict(X)
    return out


def compute_2d(tfidf: TfidfVectorizer, texts: pd.Series, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    X = tfidf.transform(texts.astype(str))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    return svd.fit_transform(X)


def top_terms_per_cluster(km: KMeans, tfidf: TfidfVectorizer, topn: int = 8) -> pd.DataFrame:
    terms = tfidf.get_feature_names_out()
    order = km.cluster_centers_.argsort()[:, ::-1]
    rows = []
    for i in range(km.n_clusters):
        words = [terms[idx] for idx in order[i, :topn]]
        rows.append({"cluster": i, "top_terms": ", ".join(words)})
    return pd.DataFrame(rows)


def compute_clustering_from_dataset(
    k: int = 6,
    maxf: int = 8000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, TfidfVectorizer]:
    ds = RecipesDataset()
    df = ds.load()

    df = df.copy()
    df["text"] = (
        df.get("name", pd.Series([""] * len(df))).fillna("")
        + " "
        + df.get("ingredients", pd.Series([""] * len(df))).astype(str)
        + " "
        + df.get("description", pd.Series([""] * len(df))).fillna("")
    ).str.lower()

    km, tfidf = build_tfidf_kmeans(df["text"], k=k, maxf=maxf, random_state=random_state)
    return df, km, tfidf
