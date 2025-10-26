"""Compute similarity between recipes using nutritional (nutriscore-like) features.

This module builds a numeric feature vector per recipe from available
nutrition columns (calories, total_fat, sugar, sodium, protein, saturated_fat,
carbohydrates) and computes a cosine-similarity matrix. It reuses
`RecipesDataset` for loading data and follows the project's helpers style.
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from core.dataset import RecipesDataset


NUTRI_COLS = [
    "calories",
    "total_fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbohydrates",
]


def _select_nutri_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric matrix (rows aligned with df) containing available nutri cols.

    Non-numeric values are coerced to NaN and rows with all-NaN are removed by the caller.
    """
    present = [c for c in NUTRI_COLS if c in df.columns]
    if not present:
        return pd.DataFrame()
    mat = df[present].apply(pd.to_numeric, errors="coerce")
    return mat


def build_nutri_similarity(df: pd.DataFrame, sample_n: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """Compute cosine similarity DataFrame using nutritional features.

    Parameters
    - df: recipes dataframe (should contain `id` and nutrition columns)
    - sample_n: optional sample size to limit computation
    - random_state: RNG seed for sampling

    Returns
    - simdf: pandas.DataFrame indexed/columned by recipe id with cosine similarity (float)
    """
    src = df.copy()
    if sample_n is not None and sample_n > 0 and len(src) > sample_n:
        src = src.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    if "id" not in src.columns:
        src = src.reset_index().rename(columns={"index": "id"})

    X = _select_nutri_matrix(src)
    if X.empty:
        return pd.DataFrame()

    # Drop rows with all-NaN in the selected nutrition columns
    valid_mask = ~X.isna().all(axis=1)
    if valid_mask.sum() == 0:
        return pd.DataFrame()

    X_valid = X[valid_mask].fillna(0.0)

    # Standardize numeric features so scales don't dominate
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_valid.values)

    sim = cosine_similarity(Xs)

    ids = src.loc[valid_mask, "id"].values
    simdf = pd.DataFrame(sim, index=ids, columns=ids)
    return simdf


def find_similar_by_nutri(sim_df: pd.DataFrame, recipe_id, top_n: int = 25) -> pd.Series:
    """Return top-N similar recipes by nutritional similarity."""
    if sim_df.empty:
        return pd.Series(dtype=float)
    if recipe_id not in sim_df.index:
        raise KeyError(f"Recipe id {recipe_id} not found in nutri similarity matrix")
    s = sim_df[recipe_id].sort_values(ascending=False)
    s = s.drop(labels=recipe_id, errors="ignore")
    return s.head(top_n)


def compute_similarity_from_dataset(anchor: str | Path | None = None, sample_n: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """Load recipes using RecipesDataset and compute nutritional similarity.

    anchor can be Path(__file__) from caller or None (defaults to module path).
    """
    anchor_path = anchor or __file__
    ds = RecipesDataset(anchor=Path(anchor_path))
    df = ds.load()
    return build_nutri_similarity(df, sample_n=sample_n, random_state=random_state)
