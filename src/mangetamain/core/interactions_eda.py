# src/core/reviews_service.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, IO
import string
from collections import Counter
import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.dataset import InteractionsDataset, DatasetLoader
from core.recipes_eda import EDAService


@dataclass
class InteractionsEDAService(EDAService):

    uploaded_file: Optional[IO[bytes]] = None  # st.file_uploader returns this
    ds: InteractionsDataset = field(default_factory=InteractionsDataset)
    
    def _load_from_file(file) -> pd.DataFrame:
        """Load a DataFrame from an uploaded file."""
        name = file.name.lower()
        if name.endswith(".parquet"):
            return pd.read_parquet(file)
        try:
            return pd.read_csv(file, engine="pyarrow")
        except Exception:
            return pd.read_csv(file)

    def load(self) -> None:
        """Load the dataset."""
        if self.uploaded_file is not None:
            self.logger.info("Loading reviews from uploaded file...")
            # Load from uploaded file
            # and set to ds.df
            self.ds.df = self._load_from_file(self.uploaded_file)
        else:
            self.ds.load()

    # ---------- Metadata ----------

    @st.cache_data(show_spinner=False)
    def duplicates(self) -> dict:
        """Get duplicate interactions.
        Returns a dict with counts of duplicates by type, or None if none found.
        Keys can include "user_recipe_date" and "full".
        """
        return_duplicates = {}
        id_duplicates = int(self.ds.df.duplicated(["user_id","recipe_id","date"]).sum())
        if id_duplicates > 0:
            return_duplicates["user_recipe_date"] = id_duplicates
            # Hashable view for duplicate detection
            df_hashable = self.ds.df.map(DatasetLoader.to_hashable)
            return_duplicates["full"] = int(sum(df_hashable.duplicated()))
            return return_duplicates
        return return_duplicates

    # ---------- Features ----------
    @st.cache_data(show_spinner=False)
    def with_text_features(self) -> pd.DataFrame:
        """Return DataFrame with basic text features extracted from reviews."""
        df = self.ds.df.copy()
        if "review" in df.columns:
            df["review_len"]      = df["review"].str.len()
            df["review_words"]    = df["review"].str.split().map(len)
            df["exclamations"]    = df["review"].str.count("!")
            df["question_marks"]  = df["review"].str.count(r"\?")
            df["has_caps"]        = df["review"].str.contains(r"[A-Z]{3,}", regex=True)
            df["mentions_thanks"] = df["review"].str.contains(r"\bthank(s| you)?\b", case=False, regex=True)
        return df

    @st.cache_data(show_spinner=False)
    def desc_numeric(self) -> pd.DataFrame:
        """Return descriptive statistics for numeric features."""
        df = self.with_text_features()
        return df.select_dtypes(include="number").describe().T

    # ---------- Aggregations ----------
    @st.cache_data(show_spinner=False)
    def agg_by_user(self, user_col: str = "user_id", rating_col: str = "rating") -> pd.DataFrame:
        """Aggregate reviews by user."""
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
        """Aggregate reviews by recipe."""
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
        """Return histogram of ratings."""
        df = self.ds.df
        if "rating" not in df.columns: 
            return pd.DataFrame()
        edges = np.linspace(0.5, 5.5, 6)  # 1..5 centrés
        counts, edges = np.histogram(df["rating"], bins=edges)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    @st.cache_data(show_spinner=False)
    def hist_review_len(self, bins: int = 50) -> pd.DataFrame:
        """Return histogram of review lengths."""
        df = self.with_text_features()
        if "review_len" not in df.columns: 
            return pd.DataFrame()
        counts, edges = np.histogram(df["review_len"], bins=bins)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    # ---------- Temporal Analysis ----------
    @st.cache_data(show_spinner=False)
    def by_month(self) -> pd.DataFrame:
        """Return DataFrame with reviews aggregated by month."""
        df = self.ds.df
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
        """Return DataFrame with seasonal profile of reviews."""
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
        """Return the range of years for which reviews are available."""
        m = self.by_month()
        if m.empty: 
            return None
        ys = m["month"].dt.year
        return int(ys.min()), int(ys.max())

    @st.cache_data(show_spinner=False)
    def one_year(self, year: int) -> pd.DataFrame:
        """Return DataFrame with reviews for a specific year."""
        m = self.by_month()
        if m.empty: 
            return m
        return m[m["month"].dt.year == year].copy()

    # ---------- Filters ----------
    @st.cache_data(show_spinner=False)
    def apply_filters(self,
                      rating_range: tuple[float,float] | None = None,
                      min_len: int = 0,
                      year: int | None = None) -> pd.DataFrame:
        """Apply filters to the reviews DataFrame."""
        df = self.with_text_features().copy()

        if rating_range and "rating" in df.columns:
            lo, hi = rating_range
            df = df[(df["rating"] >= lo) & (df["rating"] <= hi)]

        if min_len and "review_len" in df.columns:
            df = df[df["review_len"] >= int(min_len)]

        if year and "date" in df.columns:
            y = df["date"].dt.year
            df = df[y == int(year)]

        return df

    # ---------- Advanced Analysis ----------
    @st.cache_data(show_spinner=False)
    def corr_numeric(self) -> pd.DataFrame:
        """Return correlation matrix for numeric features."""
        d = self.ds.df.select_dtypes(include="number")
        if d.empty:
            return pd.DataFrame()
        return d.corr(numeric_only=True)

    @st.cache_data(show_spinner=False)
    def rating_vs_length(self) -> pd.DataFrame:
        """Return DataFrame with rating vs review length."""
        d = self.ds.df
        if not {"rating","review"}.issubset(d.columns):
            return pd.DataFrame(columns=["rating","review_len"])
        out = pd.DataFrame({
            "rating": d["rating"],
            "review_len": d["review"].fillna("").astype(str).str.len()
        }).dropna()
        return out

    @st.cache_data(show_spinner=False)
    def user_bias(self) -> pd.DataFrame:
        """Return DataFrame with user bias information."""
        d = self.ds.df
        if not {"user_id","rating"}.issubset(d.columns):
            return pd.DataFrame()
        g = (d.groupby("user_id")["rating"]
               .agg(n="size", mean="mean", median="median")
               .reset_index()
               .sort_values("n", ascending=False))
        return g

    @st.cache_data(show_spinner=False)
    def tokens_by_rating(self, k: int = 20) -> pd.DataFrame:
        """Return top k tokens by rounded rating."""
        d = self.ds.df
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
        """Export minimal clean DataFrame with key columns and types."""
        df = self.with_text_features()
        keep = [c for c in [
            "user_id", "recipe_id", "date", "rating", 
            "review", "review_len", "review_words", "exclamations"
            ] if c in df.columns]
        out = df[keep].copy()
        out = out.convert_dtypes(dtype_backend="numpy_nullable")
        return out
