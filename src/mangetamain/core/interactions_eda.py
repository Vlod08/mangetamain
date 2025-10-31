# src/core/reviews_service.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, IO
import string
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from core.dataset import InteractionsDataset, DatasetLoader
from core.recipes_eda import EDAService

# Optionnel: statsmodels pour OLS/décomposition
try:
    import statsmodels.api as sm  # noqa: F401
    from statsmodels.tsa.seasonal import seasonal_decompose

    HAS_SM = True
except Exception:
    HAS_SM = False


@dataclass
class InteractionsEDAService(EDAService):

    uploaded_file: Optional[IO[bytes]] = None  # st.file_uploader returns this
    ds: InteractionsDataset = field(default_factory=InteractionsDataset)
    text_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    label: str = "interactions"

    def _load_from_file(file) -> pd.DataFrame:
        """Load a DataFrame from an uploaded file."""
        name = file.name.lower()
        if name.endswith(".parquet"):
            return pd.read_parquet(file)
        try:
            return pd.read_csv(file, engine="pyarrow")
        except Exception:
            return pd.read_csv(file)

    def load(self, df: pd.DataFrame = None, preprocess: bool = True) -> pd.DataFrame:
        df = super().load(df, preprocess)
        self.logger.info("Computing text features ...")
        self.text_features = self.compute_text_features(self.ds.df["review"])
        return df

    # ---------- Metadata ----------
    def duplicates(self) -> dict:
        """Get duplicate interactions.
        Returns a dict with counts of duplicates by type, or None if none found.
        Keys can include "user_recipe_date" and "full".
        """
        return_duplicates = {}
        id_duplicates = int(
            self.ds.df.duplicated(["user_id", "recipe_id", "date"]).sum()
        )
        if id_duplicates > 0:
            return_duplicates["user_recipe_date"] = id_duplicates
            # Hashable view for duplicate detection
            df_hashable = self.ds.df.map(DatasetLoader.to_hashable)
            return_duplicates["full"] = int(sum(df_hashable.duplicated()))
            return return_duplicates
        return return_duplicates

    # ---------- Features ----------
    @staticmethod
    @st.cache_data(show_spinner=False)
    def compute_text_features(col: pd.Series) -> pd.DataFrame:
        """Return DataFrame with basic text features extracted from reviews."""
        df_result = pd.DataFrame()
        col_name = col.name if col.name else "review"
        df_result[f"{col_name}_len"] = col.str.len()
        df_result[f"{col_name}_words"] = col.str.split().map(len)
        df_result[f"{col_name}_exclamations"] = col.str.count("!")
        df_result[f"{col_name}_question_marks"] = col.str.count(r"\?")
        df_result[f"{col_name}_has_caps"] = col.str.contains(r"[A-Z]{3,}", regex=True)
        df_result[f"{col_name}_mentions_thanks"] = col.str.contains(
            r"\bthank(s| you)?\b", case=False, regex=True
        )
        return df_result

    def with_text_features(self) -> pd.DataFrame:
        """Return the dataset's DataFrame with basic text features extracted from reviews."""
        if self.text_features.empty:
            self.text_features = InteractionsEDAService.compute_text_features(
                self.ds.df["review"]
            )
        return pd.concat([self.ds.df, self.text_features], axis=1)

    def desc_numeric(self) -> pd.DataFrame:
        """Return descriptive statistics for numeric features."""
        df = self.with_text_features()
        return df.select_dtypes(include="number").describe().T

    # ---------- Reviews ----------
    def rating_range(self) -> tuple[float, float] | None:
        """Return the range of ratings in the dataset."""
        df = self.ds.df
        if "rating" not in df.columns:
            return None
        return float(df["rating"].min()), float(df["rating"].max())

    def review_len_range(self) -> tuple[int, int] | None:
        """Return the range of review lengths in the dataset."""
        if "review_len" not in self.text_features.columns:
            return None
        return int(self.text_features["review_len"].min()), int(
            self.text_features["review_len"].max()
        )

    # ---------- Aggregations ----------
    def agg_by_user(
        self, user_col: str = "user_id", rating_col: str = "rating"
    ) -> pd.DataFrame:
        """Aggregate reviews by user."""
        df = self.with_text_features()
        if not {user_col, rating_col}.issubset(df.columns):
            return pd.DataFrame()
        return (
            df.groupby(user_col, dropna=False)
            .agg(
                n_reviews=(rating_col, "size"),
                mean_rating=(rating_col, "mean"),
                median_rating=(rating_col, "median"),
                p95_len=("review_len", lambda s: s.dropna().quantile(0.95)),
            )
            .sort_values("n_reviews", ascending=False)
        )

    def agg_by_recipe(
        self, recipe_col: str = "recipe_id", rating_col: str = "rating"
    ) -> pd.DataFrame:
        """Aggregate reviews by recipe."""
        df = self.with_text_features()
        if not {recipe_col, rating_col}.issubset(df.columns):
            return pd.DataFrame()
        return (
            df.groupby(recipe_col)
            .agg(
                n_reviews=(rating_col, "size"),
                mean_rating=(rating_col, "mean"),
                median_rating=(rating_col, "median"),
                p95_len=("review_len", lambda s: s.quantile(0.95)),
            )
            .sort_values("n_reviews", ascending=False)
        )

    # ---------- Distributions ----------
    def hist_rating(self) -> pd.DataFrame:
        """Return histogram of ratings."""
        df = self.ds.df
        if "rating" not in df.columns:
            return pd.DataFrame()
        edges = np.linspace(0.5, 5.5, 6)  # 1..5 centrés
        counts, edges = np.histogram(df["rating"], bins=edges)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    def hist_review_len(self, bins: int = 50) -> pd.DataFrame:
        """Return histogram of review lengths."""
        df = self.with_text_features()
        if "review_len" not in df.columns:
            return pd.DataFrame()
        counts, edges = np.histogram(df["review_len"], bins=bins)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

   
    def seasonal_profile(self) -> pd.DataFrame:
        """Return DataFrame with seasonal profile of reviews."""
        m = self.by_month()
        if m.empty:
            return m
        tmp = m.assign(mon=m["month"].dt.month)
        return (
            tmp.groupby("mon")["n"]
            .mean()
            .reindex(range(1, 13))
            .reset_index()
            .rename(columns={"mon": "month", "n": "n_mean"})
        )

    def year_range(self) -> tuple[int, int] | None:
        """Return the range of years for which reviews are available."""
        m = self.by_month()
        if m.empty:
            return None
        ys = m["month"].dt.year
        return int(ys.min()), int(ys.max())

    def one_year(self, year: int) -> pd.DataFrame:
        """Return DataFrame with reviews for a specific year."""
        m = self.by_month()
        if m.empty:
            return m
        return m[m["month"].dt.year == year].copy()

    # ---------- Filters ----------
    def apply_filters(
        self,
        rating_range: tuple[float, float] | None = None,
        review_len_range: tuple[int, int] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Apply filters to the reviews DataFrame."""
        orig_columns = self.ds.df.columns.values
        df = self.with_text_features().copy()

        if rating_range and "rating" in df.columns:
            lo, hi = rating_range
            df = df[df["rating"].between(lo, hi)]

        if review_len_range and "review_len" in df.columns:
            df = df[df["review_len"].between(*review_len_range)]

        if year_range and "date" in df.columns:
            y = df["date"].dt.year
            df = df[y.between(*year_range)]

        return df[orig_columns]

    # ---------- Advanced Analysis ----------
    # def corr_numeric(self) -> pd.DataFrame:
    # """Return correlation matrix for numeric features."""
    # d = self.ds.df.select_dtypes(include="number")
    # if d.empty:
    # return pd.DataFrame()
    # return d.corr(numeric_only=True)

    def rating_vs_length(self) -> pd.DataFrame:
        """Return DataFrame with rating vs review length."""
        d = self.ds.df
        if not {"rating", "review"}.issubset(d.columns):
            return pd.DataFrame(columns=["rating", "review_len"])
        out = pd.DataFrame(
            {
                "rating": d["rating"],
                "review_len": d["review"].fillna("").astype(str).str.len(),
            }
        ).dropna()
        return out

    def user_bias(self) -> pd.DataFrame:
        """Return DataFrame with user bias information."""
        d = self.ds.df
        if not {"user_id", "rating"}.issubset(d.columns):
            return pd.DataFrame()
        g = (
            d.groupby("user_id")["rating"]
            .agg(n="size", mean="mean", median="median")
            .reset_index()
            .sort_values("n", ascending=False)
        )
        return g

    def tokens_by_rating(self, k: int = 20) -> pd.DataFrame:
        """Return top k tokens by rounded rating."""
        d = self.ds.df
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

    # ---------- Exports ----------
    def export_clean_min(self) -> pd.DataFrame:
        """Export minimal clean DataFrame with key columns and types."""
        df = self.with_text_features()
        keep = [
            c
            for c in [
                "user_id",
                "recipe_id",
                "date",
                "rating",
                "review",
                "review_len",
                "review_words",
                "exclamations",
            ]
            if c in df.columns
        ]
        out = df[keep].copy()
        out = out.convert_dtypes(dtype_backend="numpy_nullable")
        return out

        # ---------- Helpers temporels ----------

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garantit que la colonne date est en datetime64 (ou NaT)."""
        if "date" not in df.columns:
            return df
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # ---------- Temporal Analysis  ----------
    def by_month(self) -> pd.DataFrame:
        """Agrège par mois: n et rating moyen."""
        df = self._ensure_datetime(self.ds.df)
        if "date" not in df.columns or df["date"].isna().all():
            return pd.DataFrame()
        work = df.dropna(subset=["date"]).copy()
        work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
        g = work.groupby("month", as_index=False)
        out = g.agg(
            n=("date", "size"),
            mean_rating=(
                ("rating", "mean") if "rating" in work.columns else ("date", "size")
            ),
        ).sort_values("month")
        return out

    # ---------- Time Series pour la page ----------
    def monthly_series(self) -> pd.DataFrame:
        """Alias by_month avec noms attendus par la page."""
        return self.by_month()

    def monthly_rolling(self, window: int = 3) -> pd.DataFrame:
        """Ajoute des colonnes lissées (rolling 3 par défaut)."""
        bm = self.monthly_series()
        if bm.empty:
            return bm
        out = bm.copy()
        out["n_roll3"] = out["n"].rolling(window=window, min_periods=1).mean()
        if "mean_rating" in out.columns:
            out["mean_rating_roll3"] = (
                out["mean_rating"].rolling(window=window, min_periods=1).mean()
            )
        return out

    def monthly_yoy(self) -> pd.DataFrame:
        """Croissance YoY de n (t vs t-12)."""
        bm = self.monthly_series()
        if bm.empty:
            return bm
        out = bm.copy()
        out["n_prev12"] = out["n"].shift(12)
        out["n_yoy"] = (out["n"] / out["n_prev12"] - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        return out.drop(columns=["n_prev12"])

    def seasonal_decompose_monthly(self) -> pd.DataFrame:
        """Décomposition additive (trend/seasonal/resid) sur n mensuel."""
        if not HAS_SM:
            return pd.DataFrame()
        bm = self.monthly_series()
        if bm.empty or len(bm) < 24:
            return pd.DataFrame()
        s = bm.set_index("month")["n"].asfreq("MS")  # série mensuelle
        s = s.interpolate(limit_direction="both")
        dec = seasonal_decompose(
            s, model="additive", period=12, extrapolate_trend="freq"
        )
        long = (
            pd.concat(
                [
                    dec.observed.rename("observed"),
                    dec.trend.rename("trend"),
                    dec.seasonal.rename("seasonal"),
                    dec.resid.rename("resid"),
                ],
                axis=1,
            )
            .reset_index()
            .melt(id_vars="month", var_name="part", value_name="value")
            .dropna(subset=["value"])
        )
        return long

    def monthly_anomalies(self, z_thresh: float = 2.5) -> pd.DataFrame:
        """Z-score sur n mensuel pour marquer les anomalies."""
        bm = self.monthly_series()
        if bm.empty:
            return bm
        out = bm.copy()
        mu = out["n"].mean()
        sigma = out["n"].std(ddof=0)
        out["z_n"] = 0.0 if (sigma == 0 or np.isnan(sigma)) else (out["n"] - mu) / sigma
        out["is_anomaly"] = out["z_n"].abs() >= z_thresh
        return out

    def weekday_profile(self) -> pd.DataFrame:
        """Volume par jour de semaine (0=Mon..6=Sun)."""
        df = self._ensure_datetime(self.ds.df)
        if "date" not in df.columns or df["date"].isna().all():
            return pd.DataFrame()
        wk = (
            df.dropna(subset=["date"])
            .assign(wk=lambda d: d["date"].dt.weekday)
            .groupby("wk")
            .size()
            .rename("n")
            .reset_index()
            .sort_values("wk")
        )
        return wk

    def weekday_hour_heat(self) -> pd.DataFrame:
        """Matrice heure × jour (heatmap)."""
        df = self._ensure_datetime(self.ds.df)
        if "date" not in df.columns or df["date"].isna().all():
            return pd.DataFrame()
        mat = (
            df.dropna(subset=["date"])
            .assign(wk=lambda d: d["date"].dt.weekday, h=lambda d: d["date"].dt.hour)
            .groupby(["wk", "h"])
            .size()
            .rename("n")
            .reset_index()
        )
        return mat

    def cohorts_users(self) -> pd.DataFrame:
        """Cohortes par 1ère interaction (mois) et âge (mois depuis cohorte)."""
        df = self._ensure_datetime(self.ds.df)
        if (
            any(c not in df.columns for c in ["date", "user_id"])
            or df["date"].isna().all()
        ):
            return pd.DataFrame()
        d = df.dropna(subset=["date", "user_id"]).copy()
        d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
        first = d.groupby("user_id")["month"].min().rename("cohort")
        d = d.join(first, on="user_id")
        d["age"] = (d["month"].dt.year - d["cohort"].dt.year) * 12 + (
            d["month"].dt.month - d["cohort"].dt.month
        )
        out = (
            d.groupby(["cohort", "age"])
            .size()
            .rename("n")
            .reset_index()
            .sort_values(["cohort", "age"])
        )
        return out
