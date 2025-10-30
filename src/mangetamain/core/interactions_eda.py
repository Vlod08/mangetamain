# core/interactions_eda.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, IO
import string
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from core.dataset import (
    InteractionsDataset,
    INTERACTIONS_PARQUET_DATA_PATH,
)

# ----------------------------
# Helpers
# ----------------------------
def _read_uploaded(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    try:
        return pd.read_csv(uploaded, engine="pyarrow")
    except Exception:
        return pd.read_csv(uploaded)


@dataclass
class InteractionsEDAService:
    """
    EDA service for the interactions/reviews dataset.

    Loading priority:
      1. uploaded file (if provided)
      2. cleaned Parquet file in data/processed/interactions.parquet
      3. fallback to DatasetLoader (CSV or DB)
    """
    anchor: Path
    uploaded_file: Optional[IO[bytes]] = None

    @st.cache_resource(show_spinner=False)
    def _ds(self) -> InteractionsDataset:
        return InteractionsDataset()

    def load(self) -> pd.DataFrame:
        """Main loader with fallback logic."""
        if self.uploaded_file is not None:
            df = _read_uploaded(self.uploaded_file)
        elif INTERACTIONS_PARQUET_DATA_PATH.exists():
            df = pd.read_parquet(INTERACTIONS_PARQUET_DATA_PATH)
        else:
            df = self._ds().load()

        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ("user_id", "recipe_id", "rating"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "review" in df.columns:
            df["review"] = (
                df["review"]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        return df

    # ---------- BASIC QUALITY ----------
    @st.cache_data(show_spinner=False)
    def schema(self) -> pd.DataFrame:
        d = self.load()
        return pd.DataFrame({"col": d.columns, "dtype": d.dtypes.astype(str)})

    @st.cache_data(show_spinner=False)
    def na_rate(self) -> pd.Series:
        return self.load().isna().mean().sort_values(ascending=False)

    @st.cache_data(show_spinner=False)
    def cardinalities(self) -> pd.Series:
        d = self.load()

        def _to_hashable(x):
            if isinstance(x, np.ndarray):
                return tuple(x.tolist())
            if isinstance(x, (list, tuple, set, dict)):
                return str(x)
            return x

        d = d.applymap(_to_hashable)
        return d.nunique(dropna=True).sort_values(ascending=False)

    @st.cache_data(show_spinner=False)
    def duplicates(self) -> dict:
        d = self.load()
        h = d.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
        dup_total = int(h.duplicated().sum())
        keys = [c for c in ["user_id", "recipe_id", "date"] if c in d.columns]
        dup_keys = int(h.duplicated(subset=keys).sum()) if keys else None
        return {"dup_total": dup_total, "dup_on_keys": dup_keys, "keys": keys}

    # ---------- TEXT FEATURES ----------
    @st.cache_data(show_spinner=False)
    def with_text_features(self) -> pd.DataFrame:
        """Add text-derived features such as length, punctuation, etc."""
        df = self.load().copy()
        if "review" in df.columns:
            s = df["review"].fillna("")
            df["review_len"] = s.str.len()
            df["review_words"] = s.str.split().map(len)
            df["exclamations"] = s.str.count("!")
            df["question_marks"] = s.str.count(r"\?")
            df["has_caps"] = s.str.contains(r"[A-Z]{3,}", regex=True)
            df["mentions_thanks"] = s.str.contains(
                r"\bthank(s| you)?\b", case=False, regex=True
            )
        return df

    @st.cache_data(show_spinner=False)
    def desc_numeric(self) -> pd.DataFrame:
        return self.with_text_features().select_dtypes(include="number").describe().T

    # ---------- AGGREGATIONS ----------
    @st.cache_data(show_spinner=False)
    def agg_by_user(self) -> pd.DataFrame:
        df = self.with_text_features()
        if not {"user_id", "rating"}.issubset(df.columns):
            return pd.DataFrame()
        return (
            df.groupby("user_id")
            .agg(
                n=("rating", "size"),
                mean=("rating", "mean"),
                median=("rating", "median"),
                p95_len=("review_len", lambda s: s.quantile(0.95)),
            )
            .sort_values("n", ascending=False)
            .reset_index()
        )

    @st.cache_data(show_spinner=False)
    def agg_by_recipe(self) -> pd.DataFrame:
        df = self.with_text_features()
        if not {"recipe_id", "rating"}.issubset(df.columns):
            return pd.DataFrame()
        return (
            df.groupby("recipe_id")
            .agg(
                n=("rating", "size"),
                mean=("rating", "mean"),
                median=("rating", "median"),
            )
            .sort_values("n", ascending=False)
            .reset_index()
        )

    # ---------- DISTRIBUTIONS ----------
    @st.cache_data(show_spinner=False)
    def hist_rating(self) -> pd.DataFrame:
        df = self.load()
        if "rating" not in df.columns:
            return pd.DataFrame()
        s = pd.to_numeric(df["rating"], errors="coerce").dropna()
        edges = np.linspace(0.5, 5.5, 6)
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

    # ---------- TIME SERIES ----------
    @st.cache_data(show_spinner=False)
    def by_month(self) -> pd.DataFrame:
        df = self.load()
        if "date" not in df.columns:
            return pd.DataFrame()
        work = df.copy()
        work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
        agg = (
            work.groupby("month")
            .agg(
                n=("review", "size")
                if "review" in work.columns
                else ("date", "size"),
                mean_rating=("rating", "mean"),
            )
            .reset_index()
            .sort_values("month")
        )
        return agg

    # --- ADVANCED SERIES ---
    @st.cache_data(show_spinner=False)
    def monthly_series(self) -> pd.DataFrame:
        return self.by_month()

    @st.cache_data(show_spinner=False)
    def monthly_rolling(self, window: int = 3) -> pd.DataFrame:
        m = self.by_month()
        if m.empty:
            return m
        out = m.copy().sort_values("month")
        if "n" in out:
            out[f"n_roll{window}"] = out["n"].rolling(window=window, min_periods=1).mean()
        if "mean_rating" in out:
            out[f"mean_rating_roll{window}"] = out["mean_rating"].rolling(window=window, min_periods=1).mean()
        return out

    @st.cache_data(show_spinner=False)
    def monthly_yoy(self) -> pd.DataFrame:
        m = self.by_month()
        if m.empty:
            return m
        out = m.copy().sort_values("month")
        out["n_yoy"] = out["n"].pct_change(12)
        if "mean_rating" in out.columns:
            out["mean_rating_yoy"] = out["mean_rating"].pct_change(12)
        return out

    @st.cache_data(show_spinner=False)
    def monthly_anomalies(self, z_thresh: float = 2.5) -> pd.DataFrame:
        """Simple z-score anomaly detection on monthly counts."""
        from numpy import nanmean, nanstd
        m = self.by_month()
        if m.empty:
            return m
        x = m["n"].astype("float64")
        mu, sigma = nanmean(x), nanstd(x)
        if sigma == 0:
            sigma = 1.0
        z = (x - mu) / sigma
        out = m.copy()
        out["z_n"] = z
        out["is_anomaly"] = out["z_n"].abs() >= z_thresh
        return out

    @st.cache_data(show_spinner=False)
    def seasonal_decompose_monthly(
        self, model: str = "additive", period: int = 12, min_points: int = 24
    ) -> pd.DataFrame:
        """Seasonal decomposition with statsmodels (trend, seasonal, residual)."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except Exception:
            return pd.DataFrame()

        m = self.by_month()
        if m.empty or "n" not in m.columns:
            return pd.DataFrame()

        s = (
            m.set_index("month")["n"]
            .asfreq("MS")
            .astype("float64")
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
        )

        if len(s) < max(min_points, 2 * period) or s.isna().any():
            return pd.DataFrame()

        res = seasonal_decompose(s, model=model, period=period, two_sided=True, extrapolate_trend="freq")
        out = (
            pd.DataFrame({"trend": res.trend, "seasonal": res.seasonal, "resid": res.resid})
            .reset_index()
            .melt(id_vars="month", var_name="part", value_name="value")
            .dropna(subset=["value"])
        )
        return out

    # ---------- WEEKDAY/HOUR PROFILES ----------
    @st.cache_data(show_spinner=False)
    def weekday_profile(self) -> pd.DataFrame:
        d = self.load()
        if "date" not in d.columns:
            return pd.DataFrame()
        dt = pd.to_datetime(d["date"], errors="coerce")
        wk = dt.dt.weekday
        return pd.DataFrame({"wk": wk}).dropna().groupby("wk").size().reset_index(name="n")

    @st.cache_data(show_spinner=False)
    def weekday_hour_heat(self) -> pd.DataFrame:
        d = self.load()
        if "date" not in d.columns:
            return pd.DataFrame()
        dt = pd.to_datetime(d["date"], errors="coerce")
        if not hasattr(dt.dt, "hour"):
            return pd.DataFrame()
        df = pd.DataFrame({"wk": dt.dt.weekday, "h": dt.dt.hour}).dropna()
        if df.empty:
            return df
        return df.groupby(["wk", "h"]).size().reset_index(name="n")

    # ---------- COHORTS ----------
    @st.cache_data(show_spinner=False)
    def cohorts_users(self) -> pd.DataFrame:
        """User cohorts: first review month + months since first review."""
        d = self.load()
        if not {"user_id", "date"}.issubset(d.columns):
            return pd.DataFrame()
        df = d[["user_id", "date"]].dropna().copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        first = df.groupby("user_id")["month"].min().rename("cohort")
        df = df.join(first, on="user_id")
        df["age"] = (df["month"].dt.year - df["cohort"].dt.year) * 12 + (df["month"].dt.month - df["cohort"].dt.month)
        return (
            df.groupby(["cohort", "age"])
            .size()
            .reset_index(name="n")
            .sort_values(["cohort", "age"])
        )

    # ---------- USER BIAS ----------
    @st.cache_data(show_spinner=False)
    def user_bias(self) -> pd.DataFrame:
        d = self.load()
        if not {"user_id", "rating"}.issubset(d.columns):
            return pd.DataFrame()
        g = (
            d.groupby("user_id")["rating"]
            .agg(n="size", mean="mean", median="median")
            .reset_index()
            .sort_values("n", ascending=False)
        )
        return g

    # ---------- RATING VS LENGTH ----------
    @st.cache_data(show_spinner=False)
    def rating_vs_length(self) -> pd.DataFrame:
        d = self.load()
        if not {"rating", "review"}.issubset(d.columns):
            return pd.DataFrame(columns=["rating", "review_len"])
        out = pd.DataFrame(
            {"rating": pd.to_numeric(d["rating"], errors="coerce"),
             "review_len": d["review"].fillna("").astype(str).str.len()}
        ).dropna()
        return out

    # ---------- TOKENS BY RATING ----------
    @st.cache_data(show_spinner=False)
    def tokens_by_rating(self, k: int = 20) -> pd.DataFrame:
        d = self.load()
        if not {"review", "rating"}.issubset(d.columns):
            return pd.DataFrame(columns=["rating", "token", "count"])
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
        return pd.DataFrame(rows, columns=["rating", "token", "count"]).sort_values(
            ["rating", "count"], ascending=[True, False]
        )
