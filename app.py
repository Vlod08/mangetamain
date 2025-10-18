import streamlit as st
from app_utils.io import load_data
from app_utils.filters import ensure_session_filters
from app_utils.viz import hist_minutes
from app_utils.ui import use_global_ui
import pandas as pd
from app_utils.ui import use_global_ui

use_global_ui(
    page_title="Mangetamain — Recettes à l'ancienne bio",
    subtitle="Explorez Food.com : temps, ingrédients, pays, saisons et plus.",
    logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True
)


ensure_session_filters()

# ---- Données ----
df = load_data()

# ---- KPI Header ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recettes", f"{len(df):,}".replace(",", " "))  # espace insécable à la FR

# Dates (robuste aux types)
date_min = pd.to_datetime(df["submitted"]).min()
date_max = pd.to_datetime(df["submitted"]).max()
c2.metric("Période", f"{date_min.date()} → {date_max.date()}")

# Tags uniques si présent
if "tags" in df.columns:
    # tags type: liste / string "tag1,tag2" ? On gère les deux
    def _split_tags(x):
        if isinstance(x, list): return x
        if isinstance(x, str): return [t.strip() for t in x.split(",") if t.strip()]
        return []
    unique_tags = pd.Series([t for row in df["tags"].dropna().map(_split_tags) for t in row]).nunique()
    c3.metric("Tags uniques*", f"{unique_tags:,}".replace(",", " "))
else:
    c3.metric("Tags uniques*", "—")

c4.metric("Colonnes", f"{df.shape[1]}")

# ---- Aperçu ----
st.subheader("Aperçu rapide")
st.plotly_chart(hist_minutes(df), use_container_width=True)

st.info("Naviguez via le menu **Pages** (à gauche) : qualité, exploration, pays, saisons, clustering, etc.")
