"""Ingredient similarity utilities.

This module focuses only on computing recipe similarity using the
`ingredients` column. It avoids duplicating clustering logic and reuses
the dataset loader when needed.

Provided functions
- build_ingredient_similarity(df, sample_n=None, random_state=42) -> pd.DataFrame
- find_similar_by_ingredients(sim_df, recipe_id, top_n=25) -> pd.Series
- compute_similarity_from_dataset(anchor, sample_n=None, random_state=42) -> pd.DataFrame
"""

from __future__ import annotations
from typing import Optional
import ast
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.dataset import RecipesDataset


def _ings_to_plaintext(ings: object) -> str:
    """Normalize ingredients into a single lowercase space-separated string.

    Accepts lists, tuples or string representations (will attempt ast.literal_eval).
    """
    if isinstance(ings, (list, tuple)):
        return " ".join([str(t).strip().lower() for t in ings if t])
    if pd.isna(ings):
        return ""
    if isinstance(ings, str):
        try:
            v = ast.literal_eval(ings)
            if isinstance(v, (list, tuple)):
                return " ".join([str(t).strip().lower() for t in v if t])
        except Exception:
            return " ".join(
                [tok.strip().lower() for tok in ings.split() if tok.strip()]
            )
    return str(ings).strip().lower()


def build_ingredient_similarity(
    df: pd.DataFrame, sample_n: Optional[int] = None, random_state: int = 42
) -> pd.DataFrame:
    """Compute cosine similarity DataFrame using only the ingredients text.

    Parameters
    - df: recipes dataframe (must contain `id` and `ingredients` or will use index as id)
    - sample_n: optional sample size to reduce computation
    - random_state: RNG seed used for sampling

    Returns a pandas DataFrame indexed and columned by recipe id with cosine scores.
    """
    src = df.copy()
    if sample_n is not None and sample_n > 0 and len(src) > sample_n:
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
    simdf = pd.DataFrame(sim, index=src["id"].values, columns=src["id"].values)
    return simdf


def find_similar_by_ingredients(
    sim_df: pd.DataFrame, recipe_id, top_n: int = 25
) -> pd.Series:
    """Return top-N similar recipes (scores) from an ingredient-only similarity DataFrame."""
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
    anchor: str | pd.PathLike | None = None,
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load recipes via RecipesDataset and compute ingredient similarity.

    anchor: Path to use when instantiating RecipesDataset (defaults to module file path)
    """
    anchor_path = anchor or __file__
    ds = RecipesDataset(anchor=anchor_path)
    df = ds.load()
    return build_ingredient_similarity(df, sample_n=sample_n, random_state=random_state)
