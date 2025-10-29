# app/pages/recipes/country_season_page.py
from __future__ import annotations
import streamlit as st
from mangetamain.wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

from mangetamain.mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.src.mangetamain.core.recipes_eda import RecipesEDAService

MAX_TOP_N = 20

import pandas as pd
from streamlit_folium import st_folium
from src.mangetamain.core.map_builder.map import BubbleMapFolium


def app():
    use_global_ui(
        "Mangetamain —  Country & Seasonality",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True, subtitle=None, wide=True
    )

    recipes_eda_svc = RecipesEDAService()

    if 'df_country' in st.session_state:
        df = st.session_state['df_country']
    else:
        with st.spinner("Loading dataset"):
            start = time.time()
            df = recipes_eda_svc.load()
            st.write(
                f'Dataset loaded and preprocessed ({(time.time()-start)/60} min)')
            start = time.time()
            df = recipes_eda_svc.fetch_country(df)
            st.write(f'column Country added ({(time.time()-start)/60} min)')
            st.session_state['df_country'] = df

    if 'signatures' in st.session_state:
        signatures = st.session_state['signatures']
    else:
        start = time.time()
        signatures = get_all_country_signatures(df)
        st.session_state['signatures'] = signatures
        st.write(f'Signatures computed ({(time.time()-start)/60} min)')

    # -----------------------------------------------------------------
    # ÉTAPE 2 : L'INTERFACE UTILISATEUR (L'AFFICHAGE)
    # -----------------------------------------------------------------

    st.set_page_config(layout="wide")
    st.title("Analyseur de Recettes et Signatures d'Ingrédients 🧑‍🍳")

    # --- Sidebar Controls ---
    st.sidebar.header("Analysis Options")
    # Slider to select how many of the pre-calculated top ingredients to display
    top_n_to_display = st.sidebar.slider(
        "Display Top N Ingredients:",
        min_value=5,
        max_value=MAX_TOP_N,  # Should match the MAX_TOP_N used in the cached function
        value=10      # Default value
    )

    # --- Main Section: Country Signatures ---
    st.header("Signature Ingredients by Country")

    if signatures:
        # Get country list for the dropdown
        countries_list = sorted(list(signatures.keys()))

        # Country selection dropdown
        selected_country = st.selectbox(
            "Choose a country:",
            options=countries_list,
            index=None,
            placeholder="Select a country..."
        )

        # Display results if a country is selected
        if selected_country:
            # Get the sorted score dictionary for the selected country (up to Top 20)
            scores_dict_top20_sorted = signatures[selected_country]

            if scores_dict_top20_sorted:

                # --- FILTER TOP N BASED ON SLIDER ---
                # Since the dict is sorted, take the first N items based on the slider value
                # Convert items to list, slice, convert back to dict
                scores_to_display = dict(
                    list(scores_dict_top20_sorted.items())[:top_n_to_display])
                # ------------------------------------

                st.subheader(
                    f"Word Cloud for Top {top_n_to_display} Ingredients in: {selected_country}")

                if scores_to_display:
                    try:
                        # Generate Word Cloud from the filtered scores
                        wc = WordCloud(width=800,
                                       height=400,
                                       background_color='white',
                                       colormap='viridis'  # Other options: 'plasma', 'magma'
                                       ).generate_from_frequencies(scores_to_display)

                        # Display Word Cloud using matplotlib
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error generating word cloud: {e}")
                else:
                    st.info(
                        "No ingredients to display for the selected Top N value.")

            else:
                st.warning(
                    f"No signature scores found for '{selected_country}'.")
    else:
        st.error(
            "Could not compute country signatures. Check the 'get_all_country_signatures' function.")


@st.cache_data(show_spinner=False)
def get_all_country_signatures(df) -> dict:  # <-- top_n removed from arguments
    """
    Executes the costly TF-IDF calculation for countries (Top 20 max) 
    and caches the result.
    """
    # Always calculate the maximum possible for the slider
    st.write(
        f"Calculating TF-IDF for countries (Top {MAX_TOP_N})... (runs only once or if _analyzer changes)")

    signatures = RecipesEDAService.get_signatures_countries(
        df, top_n=MAX_TOP_N)
    # Corrected variable name from signatures_top20 to signatures
    st.success(
        f"Signatures (Top {MAX_TOP_N}) calculated for {len(signatures)} countries.")
    return signatures

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
    file = r"C:\Users\khali\Desktop\MS DATA-IA\Kit Big Data\mangetamain\data\recipes_raw_sample.csv"
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
