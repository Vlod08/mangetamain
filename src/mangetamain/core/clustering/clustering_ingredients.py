# src/mangetamain/core/clustering/ingredients.py
from __future__ import annotations

from typing import Optional, Any
import ast

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mangetamain.core.dataset import RecipesDataset  # <— new import


def _ings_to_plaintext(ings: Any) -> str:
    """Normalize an 'ingredients' cell into a single lowercase string."""
    if isinstance(ings, (list, tuple)):
        return " ".join(str(t).strip().lower() for t in ings if t)
    if pd.isna(ings):
        return ""
    if isinstance(ings, str):
        # try "['salt', 'pepper']"
        try:
            v = ast.literal_eval(ings)
            if isinstance(v, (list, tuple)):
                return " ".join(str(t).strip().lower() for t in v if t)
        except Exception:
            # fallback: just split spaces
            return " ".join(tok.strip().lower() for tok in ings.split() if tok.strip())
    return str(ings).strip().lower()


def build_ingredient_similarity(
    df: pd.DataFrame,
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return cosine-similarity (recipes × recipes) using ingredients only."""
    src = df.copy()

    if sample_n and sample_n > 0 and len(src) > sample_n:
        src = src.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    if "id" not in src.columns:
        src = src.reset_index().rename(columns={"index": "id"})

    src["_ings_text"] = src.get("ingredients", pd.Series([""] * len(src))).apply(
        _ings_to_plaintext
    )
    if src["_ings_text"].astype(bool).sum() == 0:
        return pd.DataFrame()

    vect = CountVectorizer(tokenizer=lambda s: s.split(" "))
    mat = vect.fit_transform(src["_ings_text"].astype(str))

    sim = cosine_similarity(mat)
    return pd.DataFrame(sim, index=src["id"].values, columns=src["id"].values)


def find_similar_by_ingredients(
    sim_df: pd.DataFrame, recipe_id: int | str, top_n: int = 25
) -> pd.Series:
    if sim_df.empty:
        return pd.Series(dtype=float)
    if recipe_id not in sim_df.index:
        raise KeyError(
            f"Recipe id {recipe_id} not found in ingredient similarity matrix"
        )
    s = sim_df[recipe_id].sort_values(ascending=False)
    s = s.drop(labels=recipe_id, errors="ignore")
    return s.head(top_n)


def compute_similarity_from_dataset(
    sample_n: Optional[int] = None, random_state: int = 42
) -> pd.DataFrame:
    """Load recipes with the app dataset service and compute ingredient similarity."""
    ds = RecipesDataset()
    df = ds.load()
    return build_ingredient_similarity(df, sample_n=sample_n, random_state=random_state)
