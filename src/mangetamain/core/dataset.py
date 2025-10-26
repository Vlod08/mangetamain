# src/core/dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, pandas as pd
import streamlit as st

def _project_root(anchor: Path) -> Path:
    p = anchor.resolve()
    for cand in [p, *p.parents]:
        if (cand / "data").is_dir():
            return cand
    return p

@dataclass
class RecipesDataset:
    anchor: Path  # __file__ de la page appelante

    @property
    def root(self) -> Path:
        return _project_root(self.anchor)

    @st.cache_data(show_spinner=False)
    def _load_files(self) -> pd.DataFrame:
        """Fichiers locaux : préfère parquet, sinon CSV."""
        pq = self.root / "data/processed/recipes_clean.parquet"
        csv = self.root / "data/raw/RAW_recipes.csv"
        if pq.exists():
            return pd.read_parquet(pq)
        if csv.exists():
            return pd.read_csv(csv)
        raise FileNotFoundError(f"Introuvable: {pq} ni {csv}")

    @st.cache_data(show_spinner=False)
    def _load_db(self) -> pd.DataFrame | None:
        """Base SQL si DATABASE_URL est définie (optionnel)."""
        url = os.getenv("DATABASE_URL")
        if not url:
            return None
        try:
            from sqlalchemy import create_engine
            eng = create_engine(url, pool_pre_ping=True)
            return pd.read_sql("SELECT * FROM recipes_clean", eng)
        except Exception:
            return None  # fallback fichiers

    def load(self) -> pd.DataFrame:
        df = self._load_db()
        if df is None:
            df = self._load_files()
        # types doux
        for c in ("minutes","n_steps","n_ingredients"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "submitted" in df.columns:
            df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")
        return df

    # --------- petits pré-calculs réutilisables ---------
    @st.cache_data(show_spinner=False)
    def unique_tags_count(self) -> int:
        df = self.load()
        if "tags" not in df.columns or df["tags"].isna().all():
            return 0
        # gère listes ou chaînes
        def _split(x):
            if isinstance(x, list): return x
            if isinstance(x, str): return [t.strip() for t in x.split(",") if t.strip()]
            return []
        return pd.Series([t for row in df["tags"].dropna().map(_split) for t in row]).nunique()


# ---- Add at the end of src/core/dataset.py ----
@dataclass
class InteractionsDataset:
    anchor: Path  # Path(__file__) from the calling page

    @property
    def root(self) -> Path:
        return _project_root(self.anchor)

    @st.cache_data(show_spinner=False)
    def _load_files(self) -> pd.DataFrame:
        pq = self.root / "data/processed/interactions_clean.parquet"
        csv = self.root / "data/raw/RAW_interactions.csv"
        if pq.exists(): return pd.read_parquet(pq)
        if csv.exists(): return pd.read_csv(csv)
        raise FileNotFoundError(f"Introuvable: {pq} ni {csv}")

    @st.cache_data(show_spinner=False)
    def _load_db(self) -> pd.DataFrame | None:
        url = os.getenv("DATABASE_URL")
        if not url: return None
        try:
            from sqlalchemy import create_engine
            eng = create_engine(url, pool_pre_ping=True)
            return pd.read_sql("SELECT * FROM interactions_clean", eng)
        except Exception:
            return None

    def load(self) -> pd.DataFrame:
        return self._load_db() or self._load_files()
