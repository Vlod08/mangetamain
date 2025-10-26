# src/main.py  (version POO-friendly + fallback)

from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- UI
from app_utils.ui import use_global_ui
use_global_ui(
    page_title="Mangetamain — Recettes à l'ancienne bio",
    subtitle="Explorez Food.com : temps, ingrédients, pays, saisons et plus.",
    logo="image/image.jpg",  
    logo_size_px=90,
    round_logo=True,
)

# --- Filtres & viz legacy
from app.app_utils.filters import ensure_session_filters
from app.app_utils.viz import hist_minutes
ensure_session_filters()

# ========= Chargement des données =========
# Essaye d'abord la POO (core/dataset.py), sinon fallback legacy load_data()
tags_count: int | None = None
try:
    # POO (si tu as créé src/core/dataset.py avec RecipesDataset)
    from core.dataset import RecipesDataset  # <-- core est directement sous src/
    ds = RecipesDataset(anchor=Path(__file__))
    df = ds.load()
    try:
        tags_count = ds.unique_tags_count()
    except Exception:
        tags_count = None
except Exception:
    # Fallback: ancien loader (rien ne casse si la POO n’est pas prête)
    from app_utils.io import load_data
    df = load_data()

# ========= KPI Header =========
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recettes", f"{len(df):,}".replace(",", " "))

date_min = pd.to_datetime(df["submitted"]).min()
date_max = pd.to_datetime(df["submitted"]).max()
c2.metric("Période", f"{date_min.date()} → {date_max.date()}")

# Tags uniques
if tags_count is None:
    # calcul compatible avec CSV (tags en str) et parquet (tags en list)
    if "tags" in df.columns:
        def _split_tags(x):
            if isinstance(x, list): return x
            if isinstance(x, str): return [t.strip() for t in x.split(",") if t.strip()]
            return []
        tags_count = pd.Series([t for row in df["tags"].dropna().map(_split_tags) for t in row]).nunique()
    else:
        tags_count = 0
c3.metric("Tags uniques*", f"{tags_count:,}".replace(",", " "))

c4.metric("Colonnes", f"{df.shape[1]}")

# ========= Aperçu =========
st.subheader("Aperçu rapide")
st.plotly_chart(hist_minutes(df), use_container_width=True)

st.info("Naviguez via le menu **Pages** (à gauche) : qualité, exploration, pays, saisons, clustering, etc.")
