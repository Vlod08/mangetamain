import streamlit as st
from app_utils.io import load_data
from app_utils.country import add_country_column
from app_utils.viz import bar_top_counts
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain —  Pays & Régions",     logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True, subtitle=None, wide=True)


#st.title("🌍 Pays & Régions")

df = add_country_column(load_data())
countries = sorted([c for c in df["country"].dropna().unique()])
if not countries:
    st.warning("Impossible d'inférer les pays depuis les tags.")
else:
    country = st.selectbox("Choisir un pays", countries)
    sub = df[df["country"] == country]
    c1, c2, c3 = st.columns(3)
    c1.metric("Recettes", f"{len(sub):,}")
    c2.metric("Minutes (médiane)", int(sub["minutes"].median()))
    c3.metric("Étapes (médiane)", int(sub["n_steps"].median()))
    # Top tags / ingrédients (comptage naïf)
    st.subheader(f"Top tags — {country}")
    st.plotly_chart(bar_top_counts(sub["tags"].astype(str).str.split(",").explode().str.strip()), use_container_width=True)
