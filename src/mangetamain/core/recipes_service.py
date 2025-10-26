# src/core/recipes_service.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from core.dataset import RecipesDataset


@dataclass
class RecipesEDAService:
    anchor: Path

    # ---------- Load ----------
    @st.cache_resource(show_spinner=False)
    def _ds(self) -> RecipesDataset:
        return RecipesDataset(anchor=self.anchor)

    @st.cache_data(show_spinner=True)
    def load(self) -> pd.DataFrame:
        return self._ds().load()

    # ---------- Data Quality ----------
    @st.cache_data(show_spinner=False)
    def schema(self) -> pd.DataFrame:
        df = self.load()
        return pd.DataFrame({"col": df.columns, "dtype": df.dtypes.astype(str)})

    @st.cache_data(show_spinner=False)
    def na_rate(self) -> pd.Series:
        return self.load().isna().mean().sort_values(ascending=False)

    @st.cache_data(show_spinner=False)
    def duplicates(self) -> dict:
        """
        Robust duplicate stats even when some columns are lists/ndarrays/dicts.
        We first build a temporary hashable view of the dataframe.
        """
        df = self.load()

        def _to_hashable(x):
            if isinstance(x, np.ndarray):
                try:
                    return tuple(_to_hashable(v) for v in x.tolist())
                except Exception:
                    return str(x)
            if isinstance(x, (list, tuple, set)):
                try:
                    return tuple(_to_hashable(v) for v in x)
                except Exception:
                    return str(x)
            if isinstance(x, dict):
                try:
                    return tuple((k, _to_hashable(v)) for k, v in sorted(x.items()))
                except Exception:
                    return str(x)
            return x  # already hashable

        # Hashable view for duplicate detection
        df_hashable = df.applymap(_to_hashable)

        dup_total = int(df_hashable.duplicated().sum())
        keys = [c for c in ["id", "name", "submitted"] if c in df_hashable.columns]
        dup_keys = int(df_hashable.duplicated(subset=keys).sum()) if keys else None

        return {"dup_total": dup_total, "dup_on_keys": dup_keys, "keys": keys}

    @st.cache_data(show_spinner=False)
    def numeric_desc(self) -> pd.DataFrame:
        return self.load().select_dtypes("number").describe().T

    @st.cache_data(show_spinner=False)
    def cardinalities(self) -> pd.Series:
        df = self.load()

        def _to_hashable(x):
            import numpy as np
            if isinstance(x, np.ndarray):
                try:
                    return tuple(_to_hashable(v) for v in x.tolist())
                except Exception:
                    return str(x)
            if isinstance(x, (list, tuple, set)):
                try:
                    return tuple(_to_hashable(v) for v in x)
                except Exception:
                    return str(x)
            if isinstance(x, dict):
                try:
                    return tuple((k, _to_hashable(v)) for k, v in sorted(x.items()))
                except Exception:
                    return str(x)
            return x

    # convert only object-like columns; numeric/datetime are fine as-is
        obj_cols = df.select_dtypes(include=["object"]).columns
        df2 = df.copy()
        for c in obj_cols:
            df2[c] = df2[c].map(_to_hashable)

        return df2.nunique(dropna=True).sort_values(ascending=False)


    # ---------- Explorer helpers ----------
    @st.cache_data(show_spinner=False)
    def minutes_hist(self, bins: int = 40) -> pd.DataFrame:
        s = pd.to_numeric(self.load().get("minutes"), errors="coerce").dropna()
        if s.empty:
            return pd.DataFrame()
        c, e = np.histogram(s, bins=bins)
        return pd.DataFrame({"left": e[:-1], "right": e[1:], "count": c})

    @st.cache_data(show_spinner=False)
    def steps_hist(self, bins: int = 25) -> pd.DataFrame:
        s = pd.to_numeric(self.load().get("n_steps"), errors="coerce").dropna()
        if s.empty:
            return pd.DataFrame()
        c, e = np.histogram(s, bins=bins)
        return pd.DataFrame({"left": e[:-1], "right": e[1:], "count": c})

    @st.cache_data(show_spinner=False)
    def by_year(self) -> pd.DataFrame:
        df = self.load().copy()
        if "submitted" not in df.columns:
            return pd.DataFrame()
        df["year"] = pd.to_datetime(df["submitted"], errors="coerce").dt.year
        return (
            df.dropna(subset=["year"])
              .groupby("year").size().reset_index(name="n")
              .sort_values("year")
        )

    @staticmethod
    def _contains_any(values: Iterable[str], items: Iterable[str]) -> bool:
        s = set(v.strip().lower() for v in values if v)
        return any(t in s for t in items)

    @st.cache_data(show_spinner=False)
    def apply_filters(
        self,
        minutes: tuple[int, int] | None = None,
        steps: tuple[int, int] | None = None,
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        include_ings: list[str] | None = None,
    ) -> pd.DataFrame:
        df = self.load().copy()

        if minutes and "minutes" in df.columns:
            lo, hi = minutes
            m = pd.to_numeric(df["minutes"], errors="coerce")
            df = df[(m >= lo) & (m <= hi)]

        if steps and "n_steps" in df.columns:
            lo, hi = steps
            s = pd.to_numeric(df["n_steps"], errors="coerce")
            df = df[(s >= lo) & (s <= hi)]

        # tags/ingredients are lists in the artefact
        if include_tags:
            inc = [t.lower().strip() for t in include_tags if t]
            df = df[df["tags"].apply(lambda lst: isinstance(lst, list) and any(t in lst for t in inc))]

        if exclude_tags:
            exc = [t.lower().strip() for t in exclude_tags if t]
            df = df[~df["tags"].apply(lambda lst: isinstance(lst, list) and any(t in lst for t in exc))]

        if include_ings:
            inc = [t.lower().strip() for t in include_ings if t]
            df = df[df["ingredients"].apply(lambda lst: isinstance(lst, list) and any(t in lst for t in inc))]

        return df

    @st.cache_data(show_spinner=False)
    def top_ingredients(self, k: int = 30) -> pd.DataFrame:
        df = self.load()
        if "ingredients" not in df.columns:
            return pd.DataFrame(columns=["ingredient", "count"])
        cnt = Counter()
        for row in df["ingredients"].dropna():
            if isinstance(row, list):
                cnt.update([str(x).lower().strip() for x in row if x])
        return pd.DataFrame(cnt.most_common(k), columns=["ingredient", "count"])
