"""Compute similarity between recipes using nutritional (nutriscore-like) features."""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from mangetamain.core.dataset import RecipesDataset  # adjust import if needed

NUTRI_COLS = [
    "calories",
    "total_fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbohydrates",
]

ALT_COL_MAP = {
    "nutrition_cal": "calories",
    "nutrition_fat": "total_fat",
    "nutrition_sugar": "sugar",
    "nutrition_protein": "protein",
    "nutrition_sodium": "sodium",
}


def _select_nutri_matrix(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in NUTRI_COLS if c in df.columns]
    if not present:
        return pd.DataFrame()
    mat = df[present].apply(pd.to_numeric, errors="coerce")
    return mat


def build_nutri_similarity(
    df: pd.DataFrame,
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    src = df.copy()

    if sample_n is not None and sample_n > 0 and len(src) > sample_n:
        src = src.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    if "id" not in src.columns:
        src = src.reset_index().rename(columns={"index": "id"})

    X = _select_nutri_matrix(src)
    if X.empty:
        return pd.DataFrame()

    valid_mask = ~X.isna().all(axis=1)
    if valid_mask.sum() == 0:
        return pd.DataFrame()

    X_valid = X[valid_mask].fillna(0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_valid.values)

    sim = cosine_similarity(Xs)
    ids = src.loc[valid_mask, "id"].values
    simdf = pd.DataFrame(sim, index=ids, columns=ids)
    return simdf


def normalize_nutrition_columns(
    df: pd.DataFrame,
    try_parse_nut_column: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Make sure df has the nutrition columns we expect.
    """
    info = {"present": [], "mapped": [], "parsed_nut_column": False, "alt_present": []}
    out = df.copy()

    # 1) map alternative names
    for alt, tgt in ALT_COL_MAP.items():
        if alt in out.columns and tgt not in out.columns:
            out[tgt] = pd.to_numeric(out[alt], errors="coerce")
            info["mapped"].append((alt, tgt))

    info["present"] = [c for c in NUTRI_COLS if c in out.columns]
    info["alt_present"] = [c for c in ALT_COL_MAP.keys() if c in out.columns]

    # 2) if we still don't have any nutrition columns, try to expand a "nutrition" column
    if try_parse_nut_column and not info["present"] and "nutrition" in out.columns:

        def _try_parse(val):
            # handle array-like directly
            if isinstance(val, (list, tuple, np.ndarray)):
                return list(val)

            # scalar NA
            if pd.api.types.is_scalar(val) and pd.isna(val):
                return None

            # string like "[1,2,3,...]"
            if isinstance(val, str):
                import ast

                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, (list, tuple, np.ndarray)):
                        return list(parsed)
                except Exception:
                    return None

            # fallback
            return None

        parsed = out["nutrition"].map(_try_parse)
        ok_mask = parsed.map(lambda x: isinstance(x, list) and len(x) >= len(NUTRI_COLS))

        if ok_mask.any():
            info["parsed_nut_column"] = True
            for i, col in enumerate(NUTRI_COLS):
                out[col] = pd.NA
                out.loc[ok_mask, col] = parsed[ok_mask].map(lambda lst, idx=i: lst[idx] if idx < len(lst) else pd.NA)

            for c in NUTRI_COLS:
                out[c] = pd.to_numeric(out[c], errors="coerce")

            info["present"] = [c for c in NUTRI_COLS if c in out.columns]

    return out, info


def build_nutri_similarity_from_raw(
    df: pd.DataFrame,
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    norm_df, info = normalize_nutrition_columns(df, try_parse_nut_column=True)
    sim = build_nutri_similarity(norm_df, sample_n=sample_n, random_state=random_state)
    return sim, info


def find_similar_by_nutri(sim_df: pd.DataFrame, recipe_id, top_n: int = 25) -> pd.Series:
    if sim_df.empty:
        return pd.Series(dtype=float)
    if recipe_id not in sim_df.index:
        raise KeyError(f"Recipe id {recipe_id} not found in nutri similarity matrix")
    s = sim_df[recipe_id].sort_values(ascending=False)
    s = s.drop(labels=recipe_id, errors="ignore")
    return s.head(top_n)


def compute_similarity_from_dataset(
    anchor: str | Path | None = None,
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    ds = RecipesDataset(anchor=anchor or __file__)
    df = ds.load()
    return build_nutri_similarity(df, sample_n=sample_n, random_state=random_state)
