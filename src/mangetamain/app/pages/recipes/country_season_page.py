# app/pages/recipes/country_season_page.py
from __future__ import annotations
import streamlit as st

from app.app_utils.viz import bar_top_counts
from app.app_utils.ui import use_global_ui
from core.recipes_eda import RecipesEDAService


def app():
    use_global_ui(
        "Mangetamain —  Country & Seasonality",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True, subtitle=None, wide=True
        )

    recipes_eda_svc = RecipesEDAService()
    df = recipes_eda_svc.fetch_country()
    countries = sorted([c for c in df["country"].dropna().unique()])
    if not countries:
        st.warning("No country information available in the dataset.")
    else:
        country = st.selectbox("Choose a country", countries)
        sub = df[df["country"] == country]
        c1, c2, c3 = st.columns(3)
        c1.metric("Recipes", f"{len(sub):,}")
        c2.metric("Minutes (median)", int(sub["minutes"].median()))
        c3.metric("Steps (median)", int(sub["n_steps"].median()))
        # Top tags / ingredients (naive counting)
        st.subheader(f"Top tags — {country}")
        st.plotly_chart(bar_top_counts(sub["tags"].astype(str).str.split(",").explode().str.strip()), width='stretch')


if __name__ == "__main__":
    app()