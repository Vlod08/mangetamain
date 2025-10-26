# src/core/reviews_service.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, IO
import string
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from core.dataset import InteractionsDataset  # dataset loader

# ----------------------------
# Helpers
# ----------------------------
_STOP_BASIC = {
    "the","a","an","and","or","is","it","to","for","of","on","in","with","this","that","these","those",
    "very","really","so","just","i","we","you","he","she","they","was","were","be","been","are","am",
    "thanks","thank","recipe"
}

def _read_uploaded(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    try:
        return pd.read_csv(uploaded, engine="pyarrow")
    except Exception:
        return pd.read_csv(uploaded)

# ----------------------------
# Service
# ----------------------------
@dataclass
class ReviewsEDAService:
    anchor: Path                       # Path(__file__) from the page
    uploaded_file: Optional[IO[bytes]] = None  # st.file_uploader returns this

    # --- data source (cached resource) ---
    @st.cache_resource(show_spinner=False)
    def _ds(self) -> InteractionsDataset:
        return InteractionsDataset(anchor=self.anchor)

    # --- split default vs uploaded to avoid caching issues with UploadedFile ---
    @st.cache_data(show_spinner=True)
    def _load_default(self) -> pd.DataFrame:
        """Cached load from dataset (files/DB)."""
        return self._ds().load()

    def _load_uploaded(self) -> pd.DataFrame:
        """Non-cached load for uploaded file (unhashable object)."""
        return _read_uploaded(self.uploaded_file)

    def load(self) -> pd.DataFrame:
        df = self._load_uploaded() if self.uploaded_file is not None else self._load_default()

        # normalize
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("user_id","recipe_id","rating"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "review" in df.columns:
            df["review"] = (
                df["review"].astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        return df

    # ---------- Qualité ----------
    @st.cache_data(show_spinner=False)
    def schema(self) -> pd.DataFrame:
        d = self.load()
        return pd.DataFrame({"col": d.columns, "dtype": d.dtypes.astype(str)})

    @staticmethod
    def _to_hashable(x):
        if isinstance(x, np.ndarray):
            try:
                return tuple(ReviewsEDAService._to_hashable(v) for v in x.tolist())
            except Exception:
                return str(x)
        if isinstance(x, (list, tuple, set)):
            try:
                return tuple(ReviewsEDAService._to_hashable(v) for v in x)
            except Exception:
                return str(x)
        if isinstance(x, dict):
            try:
                return tuple((k, ReviewsEDAService._to_hashable(v)) for k, v in sorted(x.items()))
            except Exception:
                return str(x)
        return x

    @st.cache_data(show_spinner=False)
    def na_rate(self) -> pd.Series:
        return self.load().isna().mean().sort_values(ascending=False)

    @st.cache_data(show_spinner=False)
    def cardinalities(self) -> pd.Series:
        d = self.load().applymap(self._to_hashable)
        return d.nunique(dropna=True).sort_values(ascending=False)

    @st.cache_data(show_spinner=False)
    def duplicates(self) -> dict:
        d = self.load().applymap(self._to_hashable)
        dup_total = int(d.duplicated().sum())
        keys = [c for c in ["user_id","recipe_id","date"] if c in d.columns]
        dup_keys = int(d.duplicated(subset=keys).sum()) if keys else None
        return {"dup_total": dup_total, "dup_on_keys": dup_keys, "keys": keys}

    # ---------- Features ----------
    @st.cache_data(show_spinner=False)
    def with_text_features(self) -> pd.DataFrame:
        df = self.load().copy()
        if "review" in df.columns:
            s = df["review"].fillna("")
            df["review_len"]     = s.str.len()
            df["review_words"]   = s.str.split().map(len)
            df["exclamations"]   = s.str.count("!")
            df["question_marks"] = s.str.count(r"\?")
            df["has_caps"]       = s.str.contains(r"[A-Z]{3,}", regex=True)
            df["mentions_thanks"]= s.str.contains(r"\bthank(s| you)?\b", case=False, regex=True)
        return df

    @st.cache_data(show_spinner=False)
    def desc_numeric(self) -> pd.DataFrame:
        df = self.with_text_features()
        return df.select_dtypes(include="number").describe().T

    # ---------- Agrégations ----------
    @st.cache_data(show_spinner=False)
    def agg_by_user(self, user_col: str = "user_id", rating_col: str = "rating") -> pd.DataFrame:
        df = self.with_text_features()
        if not {user_col, rating_col}.issubset(df.columns): 
            return pd.DataFrame()
        return (df.groupby(user_col)
                .agg(n_reviews=(rating_col,"size"),
                     mean_rating=(rating_col,"mean"),
                     median_rating=(rating_col,"median"),
                     p95_len=("review_len", lambda s: s.quantile(0.95) if s is not None else np.nan))
                .sort_values("n_reviews", ascending=False))

    @st.cache_data(show_spinner=False)
    def agg_by_recipe(self, recipe_col: str = "recipe_id", rating_col: str = "rating") -> pd.DataFrame:
        df = self.with_text_features()
        if not {recipe_col, rating_col}.issubset(df.columns): 
            return pd.DataFrame()
        return (df.groupby(recipe_col)
                .agg(n_reviews=(rating_col,"size"),
                     mean_rating=(rating_col,"mean"),
                     median_rating=(rating_col,"median"),
                     p95_len=("review_len", lambda s: s.quantile(0.95) if s is not None else np.nan))
                .sort_values("n_reviews", ascending=False))

    # ---------- Distributions ----------
    @st.cache_data(show_spinner=False)
    def hist_rating(self) -> pd.DataFrame:
        df = self.load()
        if "rating" not in df.columns: 
            return pd.DataFrame()
        s = pd.to_numeric(df["rating"], errors="coerce").dropna()
        edges = np.linspace(0.5, 5.5, 6)  # 1..5 centrés
        counts, edges = np.histogram(s, bins=edges)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    @st.cache_data(show_spinner=False)
    def hist_review_len(self, bins: int = 50) -> pd.DataFrame:
        df = self.with_text_features()
        if "review_len" not in df.columns: 
            return pd.DataFrame()
        s = pd.to_numeric(df["review_len"], errors="coerce").dropna()
        counts, edges = np.histogram(s, bins=bins)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    # ---------- Temporalité ----------
    @st.cache_data(show_spinner=False)
    def by_month(self) -> pd.DataFrame:
        df = self.load()
        if "date" not in df.columns: 
            return pd.DataFrame()
        work = df.copy()
        work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
        agg = (work.groupby("month")
               .agg(n=("review","size") if "review" in work.columns else ("date","size"),
                    mean_rating=("rating","mean"))
               .reset_index()
               .sort_values("month"))
        return agg

    @st.cache_data(show_spinner=False)
    def seasonal_profile(self) -> pd.DataFrame:
        m = self.by_month()
        if m.empty: 
            return m
        tmp = m.assign(mon=m["month"].dt.month)
        return (tmp.groupby("mon")["n"].mean()
                .reindex(range(1,13))
                .reset_index()
                .rename(columns={"mon":"month", "n":"n_mean"}))

    @st.cache_data(show_spinner=False)
    def year_range(self) -> tuple[int, int] | None:
        m = self.by_month()
        if m.empty: 
            return None
        ys = m["month"].dt.year
        return int(ys.min()), int(ys.max())

    @st.cache_data(show_spinner=False)
    def one_year(self, year: int) -> pd.DataFrame:
        m = self.by_month()
        if m.empty: 
            return m
        return m[m["month"].dt.year == year].copy()

    # ---------- Filtres tableau ----------
    @st.cache_data(show_spinner=False)
    def apply_filters(self,
                      rating_range: tuple[float,float] | None = None,
                      min_len: int = 0,
                      year: int | None = None) -> pd.DataFrame:
        df = self.with_text_features().copy()

        if rating_range and "rating" in df.columns:
            lo, hi = rating_range
            r = pd.to_numeric(df["rating"], errors="coerce")
            df = df[(r >= lo) & (r <= hi)]

        if min_len and "review_len" in df.columns:
            df = df[df["review_len"] >= int(min_len)]

        if year and "date" in df.columns:
            y = pd.to_datetime(df["date"], errors="coerce").dt.year
            df = df[y == int(year)]

        return df

    # ---------- Analyse avancée ----------
    @st.cache_data(show_spinner=False)
    def corr_numeric(self) -> pd.DataFrame:
        d = self.load().select_dtypes(include="number")
        if d.empty:
            return pd.DataFrame()
        return d.corr(numeric_only=True)

    @st.cache_data(show_spinner=False)
    def rating_vs_length(self) -> pd.DataFrame:
        d = self.load()
        if not {"rating","review"}.issubset(d.columns):
            return pd.DataFrame(columns=["rating","review_len"])
        out = pd.DataFrame({
            "rating": pd.to_numeric(d["rating"], errors="coerce"),
            "review_len": d["review"].fillna("").astype(str).str.len()
        }).dropna()
        return out

    @st.cache_data(show_spinner=False)
    def user_bias(self) -> pd.DataFrame:
        d = self.load()
        if not {"user_id","rating"}.issubset(d.columns):
            return pd.DataFrame()
        g = (d.groupby("user_id")["rating"]
               .agg(n="size", mean="mean", median="median")
               .reset_index()
               .sort_values("n", ascending=False))
        return g

    @st.cache_data(show_spinner=False)
    def tokens_by_rating(self, k: int = 20) -> pd.DataFrame:
        d = self.load()
        if not {"review","rating"}.issubset(d.columns):
            return pd.DataFrame(columns=["rating","token","count"])
        tbl = str.maketrans("", "", string.punctuation.replace("'", ""))
        d = d.dropna(subset=["rating"]).copy()
        d["bucket"] = pd.to_numeric(d["rating"], errors="coerce").round()
        rows = []
        for r, grp in d.groupby("bucket"):
            cnt = Counter()
            for txt in grp["review"].fillna("").str.lower().str.translate(tbl):
                for t in txt.split():
                    if t:
                        cnt.update([t])
            for tok, c in cnt.most_common(k):
                rows.append((int(r), tok, int(c)))
        return (pd.DataFrame(rows, columns=["rating","token","count"])
                  .sort_values(["rating","count"], ascending=[True, False]))

    # ---------- Exports ----------
    @st.cache_data(show_spinner=False)
    def export_clean_min(self) -> pd.DataFrame:
        df = self.with_text_features()
        keep = [c for c in ["user_id","recipe_id","date","rating","review","review_len","review_words","exclamations"] if c in df.columns]
        out = df[keep].copy()
        out = out.convert_dtypes(dtype_backend="numpy_nullable")
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out
