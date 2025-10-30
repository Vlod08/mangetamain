# app/pages/recipes/country_season_page.py
from __future__ import annotations
import streamlit as st
from streamlit_lottie import st_lottie
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import pandas as pd

from mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.core.recipes_eda import RecipesEDAService
from mangetamain.core.utils.utils import load_lottie


MIN_TOP_N = 5
MAX_TOP_N = 20

def format_time(start_time: float, end_time: float) -> str:
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed, 60)
    if minutes == 0:
        return f"{seconds:.2f} sec"
    return f"{int(minutes)} min {seconds:.2f} sec"

def fetch_country_animation(recipes_eda_svc: RecipesEDAService, df: pd.DataFrame):
    # Placeholders
    lottie_placeholder = st.empty()

    lottie = load_lottie()
    with lottie_placeholder:
        st.write("Fetching country column...")
        st_lottie(lottie, height=200, speed=1, loop=True)

    start_time = time.time()
    df_country = recipes_eda_svc.fetch_country(df)
    end_time = time.time()

    formatted = format_time(start_time, end_time)

    # --- Replace loading elements with success message ---
    lottie_placeholder.empty()
    st.toast(
        f"Country column added ({formatted})", 
        icon=":material/thumb_up:", 
        duration=5)
    
    return df_country

def display_signature(
        signature: dict, 
        selected_country: str, 
        top_n_to_display: int):
    """Displays the signature of the selected country as a word cloud
    Args:
        signature (dict): The signature of the country containing ingredient frequencies.
        selected_country (str): The country to use for display.
        top_n_to_display (int): The number of top ingredients to display.
    """

    if not signature:
        st.warning(f"No signature scores found for '{selected_country}'.")
        return

    # --- FILTER TOP N ---
    scores_to_display = dict(list(signature.items())[:top_n_to_display])

    st.subheader(
        f"Word Cloud for Top {top_n_to_display} Ingredients in {selected_country.title()}"
    )

    if not scores_to_display:
        st.info("No ingredients to display for the selected Top N value.")
        return

    try:
        # --- DYNAMIC WIDTH BASED ON STREAMLIT CONTAINER ---
        # The more words, the taller the cloud (but cap it for very large numbers)
        base_height = 300
        height = min(800, base_height + top_n_to_display * 5)
        width = 1200

        # --- GENERATE WORD CLOUD ---
        wc = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap='viridis',
            prefer_horizontal=0.9,
            random_state=42
        ).generate_from_frequencies(scores_to_display)

        # --- DISPLAY WITH RESPONSIVE FIGURE SIZE ---
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")

        st.pyplot(fig, width='stretch')

    except Exception as e:
        st.error(f"Error generating word cloud: {e}")

def app():
    use_global_ui(
        "Mangetamain —  Country & Seasonality",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True, subtitle=None, wide=True
    )

    # Recipes dataset already uploaded in app entrypoint (main.py)
    recipes_df = st.session_state["recipes"]
    recipes_eda_svc = RecipesEDAService()
    recipes_eda_svc.load(recipes_df, preprocess=False)

    if 'df_country' in st.session_state:
        df_country = st.session_state['df_country']
    else:
        df_country = fetch_country_animation(recipes_eda_svc, recipes_df[:50])
        st.session_state['df_country'] = df_country

    countries_list = df_country["country"].dropna().sort_values().unique().tolist()

    if "signatures" in st.session_state:
        signatures = st.session_state["signatures"]
    else:
        start = time.time()
        signatures = RecipesEDAService.get_signatures_countries(
            df_country, top_n=MAX_TOP_N)
        if not signatures:
            st.error("Could not compute country signatures !")
            return
        # st.success(f"Signatures (Top {MAX_TOP_N}) calculated for \
        #            {len(signatures)} countries.")
        formatted = format_time(start, time.time())
        st.session_state["signatures"] = signatures
        st.toast(
            f"Signatures computed ({formatted})", 
            icon=":material/thumb_up:", 
            duration=5)

    assert all(country in countries_list for country in list(signatures.keys())), \
        "Signatures list of countries does not match the list of countries in the dataset"

    # UI
    st.title("🧑‍🍳 Recipes and Ingredients Signatures Analyzer")

    # --- Sidebar Controls ---
    st.sidebar.header("Analysis Options")
    # Slider to select how many of the pre-calculated top ingredients to display
    top_n_to_display = st.sidebar.slider(
        "Display Top N Ingredients:",
        min_value=MIN_TOP_N,
        max_value=MAX_TOP_N,
        value=MAX_TOP_N
    )

    # --- Main Section: Country Signatures ---
    st.header("Signature Ingredients by Country")

    # Country selection dropdown
    selected_country = st.selectbox(
        "Select a country",
        options=countries_list,
        index=None,
        placeholder="Select a country...", 
        label_visibility='hidden'
    )

    # Display results if a country is selected
    if selected_country:
        display_signature(
            signatures[selected_country], 
            selected_country, 
            top_n_to_display)

if __name__ == "__main__":
    app()
