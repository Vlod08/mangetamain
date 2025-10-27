"""Clustering utilities for recipes (TF-IDF + KMeans).

This module extracts the core clustering logic from the Streamlit page so the
UI layer remains thin. Exported helpers mirror the style of
`core/clustering_ingredients.py`:

- build_tfidf_kmeans(corpus, k, maxf, random_state) -> (KMeans, TfidfVectorizer)
- assign_clusters(df, tfidf, km, text_col='text') -> pd.DataFrame (with 'cluster')
- compute_2d(tfidf, texts, n_components=2, random_state=42) -> np.ndarray
- top_terms_per_cluster(km, tfidf, topn=8) -> pd.DataFrame
- compute_clustering_from_dataset(anchor=None, k=6, maxf=8000) -> (df, km, tfidf)
"""
from __future__ import annotations
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from core.dataset import RecipesDataset


def build_tfidf_kmeans(corpus: pd.Series, k: int = 6, maxf: int = 8000, random_state: int = 42) -> Tuple[KMeans, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer and KMeans on the provided text corpus.

    Parameters
    - corpus: pandas Series of text strings
    - k: number of clusters
    - maxf: max_features for TfidfVectorizer

    Returns (km, tfidf) where km is the fitted KMeans and tfidf the fitted vectorizer.
    """
    tfidf = TfidfVectorizer(max_features=maxf, stop_words="english")
    X = tfidf.fit_transform(corpus.astype(str))
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(X)
    return km, tfidf


def assign_clusters(df: pd.DataFrame, tfidf: TfidfVectorizer, km: KMeans, text_col: str = "text") -> pd.DataFrame:
    """Return a copy of `df` with a `cluster` integer column assigned using `tfidf`+`km`.

    If the text column is missing it will be created as empty strings.
    """
    out = df.copy()
    if text_col not in out.columns:
        out[text_col] = ""
    X = tfidf.transform(out[text_col].astype(str))
    labels = km.predict(X)
    out["cluster"] = labels
    return out


def compute_2d(tfidf: TfidfVectorizer, texts: pd.Series, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Compute a quick 2D projection (TruncatedSVD) from texts via tfidf.transform.

    Returns an (N, 2) numpy array.
    """
    X = tfidf.transform(texts.astype(str))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    XY = svd.fit_transform(X)
    return XY


def top_terms_per_cluster(km: KMeans, tfidf: TfidfVectorizer, topn: int = 8) -> pd.DataFrame:
    """Return a DataFrame with top terms per cluster (cluster, top_terms).
    """
    terms = tfidf.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    rows = []
    for i in range(km.n_clusters):
        words = [terms[ind] for ind in order_centroids[i, :topn]]
        rows.append({"cluster": i, "top_terms": ", ".join(words)})
    return pd.DataFrame(rows)


def compute_clustering_from_dataset(anchor: Optional[str] = None, k: int = 6, maxf: int = 8000, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans, TfidfVectorizer]:
    """Load recipes using RecipesDataset and compute clustering.

    Returns (df, km, tfidf) where `df` is the loaded recipes DataFrame (unchanged),
    `km` and `tfidf` are fitted objects.
    """
    ds = RecipesDataset(anchor=anchor or __file__)
    df = ds.load()
    # build a default 'text' column like the UI did (name + ingredients + description)
    df = df.copy()
    df["text"] = (df.get("name", pd.Series([""] * len(df))).fillna("")
                  + " " + df.get("ingredients", pd.Series([""] * len(df))).astype(str)
                  + " " + df.get("description", pd.Series([""] * len(df))).fillna("")).str.lower()
    km, tfidf = build_tfidf_kmeans(df["text"], k=k, maxf=maxf, random_state=random_state)
    return df, km, tfidf
