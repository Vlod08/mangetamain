# app/pages/recipes/country_season_page.py
from __future__ import annotations
import streamlit as st

from app.app_utils.viz import bar_top_counts
from app.app_utils.ui import use_global_ui
from core.recipes_eda import RecipesEDAService

import pandas as pd
from streamlit_folium import st_folium
from map import BubbleMapFolium


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

    #  Bubble map of recipes by country/continent with click-to-filter
    st.set_page_config(page_title="Recipes Bubble Map", layout="wide")
    st.title("Recipes Bubble Map")

    # APPLY PENDING CLICK BEFORE WIDGETS 
    # If a previous click stored a pending choice, apply it now (before selectbox exists)
    if "__pending_country_choice" in st.session_state:
        st.session_state["country_choice"] = st.session_state.pop("__pending_country_choice")

    # Init state if first run
    if "country_choice" not in st.session_state:
        st.session_state["country_choice"] = "(aucun)"

    # DATA LOAD
    file = r"C:\Users\khali\Desktop\MS DATA-IA\Kit Big Data\mangetamain\data\raw_data\recipes_raw_sample.csv"
    df_raw = pd.read_csv(file)

    # LEVEL SELECTION
    level = st.radio("View", ["country", "continent"], horizontal=True)

    mapper = BubbleMapFolium(tiles="OpenStreetMap", auto_centroids=True)
    if level == "country":
        counts = mapper.counts_by_country(df_raw, country_col="country")
    else:
        counts = mapper.counts_by_continent(df_raw, continent_col="continent")

    # COUNTRY SELECTION WIDGET
    selected_country = None
    country_list = []
    if level == "country":
        country_list = counts["country"].dropna().astype(str).sort_values().unique().tolist()
        options = ["(aucun)"] + country_list
        # index comes from session state
        try:
            idx = options.index(st.session_state["country_choice"])
        except ValueError:
            idx = 0

        # Bind the selectbox to the same key we control via session_state
        st.selectbox("Choisir un pays à surligner", options, index=idx, key="country_choice")
        selected_country = None if st.session_state["country_choice"] == "(aucun)" else st.session_state["country_choice"]

    # Build map
    m = mapper.build_map(
        counts,
        level=level,
        min_px=6,
        max_px=6,
        opacity=0.95,
        use_sqrt=True,
        cluster=False,
        color_polygons=True,
        selected_country=selected_country,
        zoom_on_selected=True,
        zoom_mode="country",
    )

    # Show map
    result = st_folium(m, height=640, width=None, key="map_widget")

    # CLICK HANDLER (COUNTRY SELECTION)
    if level == "country":
        clicked_popup = (result or {}).get("last_object_clicked_popup")
        if clicked_popup:
            raw = str(clicked_popup).strip()
            # handle "<b>France</b><br/>…" or plain "France"
            if "<b>" in raw and "</b>" in raw:
                start = raw.find("<b>") + 3
                end = raw.find("</b>", start)
                country_clicked = raw[start:end].strip()
            else:
                country_clicked = raw.split("<br")[0].strip()

            if country_clicked in country_list and country_clicked != st.session_state.get("country_choice"):
                # Store as pending and rerun; next run will set country_choice before widget creation
                st.session_state["__pending_country_choice"] = country_clicked
                st.rerun()



if __name__ == "__main__":
    app()

