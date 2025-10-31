# app/pages/recipes/country_season_page.py
from __future__ import annotations
import streamlit as st
from streamlit_lottie import st_lottie
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import pandas as pd
import plotly.express as px

from mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.core.recipes_eda import RecipesEDAService
from mangetamain.core.utils.utils import load_lottie


MIN_TOP_N = 5
MAX_TOP_N = 50


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


def fetch_period_animation(recipes_eda_svc: RecipesEDAService, df: pd.DataFrame):
    # Placeholders
    lottie_placeholder = st.empty()

    lottie = load_lottie()
    with lottie_placeholder:
        st.write("Fetching saeson column...")
        st_lottie(lottie, height=200, speed=1, loop=True)

    start_time = time.time()
    df_period = recipes_eda_svc.fetch_period(df)
    end_time = time.time()

    formatted = format_time(start_time, end_time)

    # --- Replace loading elements with success message ---
    lottie_placeholder.empty()
    st.toast(
        f"Season and Event columns added ({formatted})",
        icon=":material/thumb_up:",
        duration=5)

    return df_period


def display_signature(
        signature: dict,
        selected_feature: str,
        top_n_to_display: int,
        country: bool,
        season: bool = False):
    """Displays the signature of the selected feature as a word cloud
    Args:
        signature (dict): The signature of the feature containing ingredient frequencies.
        selected_feature (str): The feature to use for display.
        top_n_to_display (int): The number of top ingredients to display.
    """

    if not signature:
        st.warning(f"No signature scores found for '{selected_feature}'.")
        return

    # --- FILTER TOP N ---
    scores_to_display = dict(list(signature.items())[:top_n_to_display])

    if not scores_to_display:
        st.info("No ingredients to display for the selected Top N value.")
        return

    st.subheader(
        f"Word Cloud for Top {top_n_to_display} Signatures in {selected_feature.title()}"
    )

    base_height = 300
    height = min(800, base_height + top_n_to_display * 5)
    width = 1200

    try:
        if country:
            st.markdown(f"""
            This cloud visualizes the ingredients that are most **characteristic** of **{selected_feature.title()}**.

            The size of each ingredient is determined by its **TF-IDF score**, not just its raw frequency. This score highlights ingredients that are not only *frequently used* in this country but also *relatively unique* to it compared to all other cuisines.

            This is what defines a **"culinary signature"**: it filters out globally common items (like 'salt' or 'water') to reveal the ingredients that truly make this cuisine distinct.
            """)
            wc = WordCloud(
                width=width,
                height=height,
                background_color='white',
                colormap='ocean',
                prefer_horizontal=0.95,
                random_state=42, margin=5
            ).generate_from_frequencies(scores_to_display)

        elif season:
            st.markdown("""
            This analysis shows which ingredients are strongly associated with a specific season (like 'pumpkin' in Fall or 'basil' in Summer).
            """)
            wc = WordCloud(
                width=width,
                height=height,
                background_color='white',
                colormap='plasma',
                prefer_horizontal=0.95,
                random_state=42, margin=5
            ).generate_from_frequencies(scores_to_display)

    # try:
        # --- DYNAMIC WIDTH BASED ON STREAMLIT CONTAINER ---
        # The more words, the taller the cloud (but cap it for very large numbers)
        # base_height = 300
        # height = min(800, base_height + top_n_to_display * 5)
        # width = 1200

        # # --- GENERATE WORD CLOUD ---
        # wc = WordCloud(
        #     width=width,
        #     height=height,
        #     background_color='white',
        #     colormap='ocean',
        #     prefer_horizontal=0.95,
        #     random_state=42, margin=5
        # ).generate_from_frequencies(scores_to_display)

        # --- DISPLAY WITH RESPONSIVE FIGURE SIZE ---
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")

        st.pyplot(fig, width='stretch')

    except Exception as e:
        st.error(f"Error generating word cloud: {e}")


def display_signatures_tfidf_vs_tf(
        signatures_tfidf: dict,
        signatures_tf: dict,
        selected_country: str,
        top_n_to_display: int):
    """Displays an interactive scatter plot (TF vs TF-IDF) for the selected country.

    Args:
        signatures_tfidf (dict): Dictionary of TF-IDF scores by country ({country: {term: score}}).
        signatures_tf (dict): Dictionary of TF scores by country.
        selected_country (str): The country to use for the display.
        top_n_to_display (int): The number of top terms to display.
    """

    if selected_country not in signatures_tfidf or not signatures_tfidf[selected_country]:
        st.warning(f"No scores found for '{selected_country}'.")
        return

    # 1. --- DATA PREPARATION ---
    # Convert data for the selected country into a DataFrame

    # Terms sorted by TF-IDF for the country
    country_tfidf = signatures_tfidf[selected_country]

    # Sort by descending TF-IDF
    sorted_terms = sorted(country_tfidf.items(),
                          key=lambda item: item[1], reverse=True)

    # Select the Top N terms
    top_scores_tfidf = dict(sorted_terms[:top_n_to_display])
    top_terms = top_scores_tfidf.keys()

    # Create data lists for the DataFrame
    data = {
        'Term': list(top_terms),
        'TFIDF': [float(top_scores_tfidf[term]) for term in top_terms],
        # Retrieve the corresponding TF scores
        'TF': [float(signatures_tf[selected_country].get(term, 0)) for term in top_terms]
    }

    df_plot = pd.DataFrame(data)

    # 2. --- TITLE DISPLAY AND CHECK ---
    st.subheader(
        f"TF vs TF-IDF Analysis for Top {top_n_to_display} Signatures in **{selected_country.title()}**"
    )

    if df_plot.empty:
        st.info("No terms to display for the selected Top N value.")
        return

    # 3. --- PLOTLY GRAPH GENERATION ---
    try:
        fig = px.scatter(
            df_plot,
            x='TF',
            y='TFIDF',
            hover_name='Term',  # Label points on hover
            size='TFIDF',       # Point size based on TF-IDF
            color='TFIDF',      # Color based on TF-IDF
            log_x=False,
            labels={
                'TF': 'Term Frequency (TF)', 'TFIDF': 'Importance (TF-IDF)'}
        )

        fig.update_traces(textposition='top center')

        # 4. --- STREAMLIT DISPLAY ---
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating Plotly chart: {e}")


# def display_seasonal_heatmap(signatures_tfidf: dict, top_n: int = 50):
#     """
#     Affiche une heatmap comparant les scores TF-IDF des ingrédients
#     pour toutes les saisons.

#     Args:
#         signatures_tfidf (dict): Le dict des scores TF-IDF ({season: {term: score}}).
#         top_n (int): Le nombre total d'ingrédients à afficher
#                      (basé sur leur meilleur score toutes saisons confondues).
#     """
#     st.header("Seasonal Signature Heatmap")

#     df_wide = pd.DataFrame.from_dict(signatures_tfidf).fillna(0)
#     df_wide['max_score'] = df_wide.max(axis=1)
#     df_top = df_wide.sort_values(by='max_score', ascending=False).head(top_n)
#     df_plot = df_top.drop(columns=['max_score'])

#     if df_plot.empty:
#         st.warning("No data to display in heatmap.")
#         return

#     # 3. --- Affichage Plotly ---
#     fig = px.imshow(
#         df_plot,
#         labels=dict(x="Season", y="Ingredient", color="TF-IDF Score"),
#         title="Top Ingredient Signatures by Season",
#         color_continuous_scale='OrRd',  # Thème de couleur (Orange-Red)
#         aspect="auto"  # S'adapte à la taille
#     )

#     # Ajuste la hauteur en fonction du nombre d'ingrédients
#     fig.update_layout(height=max(400, top_n * 20))
#     st.plotly_chart(fig, use_container_width=True)

#     with st.expander("How to read this chart?"):
#         st.markdown("""
#         This heatmap shows the **relative importance (TF-IDF score)** of an ingredient (row) for each season (column).

#         * A **dark red** cell means the ingredient is a *strong signature* for that season.
#         * A **light yellow** cell means the score is low or zero.

#         This helps you instantly spot seasonal patterns, like "Cinnamon" being high in "Fall" but low in "Summer".
#         """)


def display_seasonal_pie(df_period: pd.DataFrame):
    st.subheader("Recipe Distribution by Season")
    seasons_counts = RecipesEDAService.count_recipes_seasons(df_period)
    if seasons_counts.empty:
        st.info("No seasonal data to display.")
        return
    color_map = {
        'winter': '#AEC6CF',  # Bleu pastel
        'spring': '#B7E4C7',  # Vert menthe
        'summer': '#FFFACD',  # Jaune pâle (citron)
        'fall':   '#FFDAB9'   # Orange pâle (pêche)
    }

    fig = px.pie(
        seasons_counts,
        names='season',
        values='number of recipes',
        title='Recipe Distribution by Season',
        hole=.3,
        color='season',
        color_discrete_map=color_map
    )

    fig.update_traces(
        textposition='outside',
        textinfo='percent+label',
        pull=[0.05, 0.05, 0.05, 0.05]
    )

    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


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
        df_country = fetch_country_animation(
            recipes_eda_svc, recipes_df)
        st.session_state['df_country'] = df_country

    if 'df_period' in st.session_state:
        df_period = st.session_state['df_period']
    else:
        df_period = fetch_period_animation(
            recipes_eda_svc, recipes_df)
        st.session_state['df_period'] = df_period

    countries_list = df_country["country"].dropna(
    ).sort_values().unique().tolist()

    seasons_list = df_period["season"].dropna().sort_values().unique().tolist()

    # Signatures by country
    if "signatures_country" in st.session_state:
        signatures_country = st.session_state["signatures_country"]
    else:
        start = time.time()
        signatures_country = RecipesEDAService.get_signatures_countries(
            df_country, top_n=MAX_TOP_N)
        if not signatures_country:
            st.error("Could not compute country signatures !")
            return
        formatted = format_time(start, time.time())
        st.session_state["signatures_country"] = signatures_country
        st.toast(
            f"Signatures_country computed ({formatted})",
            icon=":material/thumb_up:",
            duration=5)

    assert all(country in countries_list for country in list(signatures_country[0].keys())), (
        "Signatures list of countries does not match the list of countries in the dataset")

    # Signatures by season
    if "signatures_season" in st.session_state:
        signatures_season = st.session_state["signatures_season"]
    else:
        start = time.time()
        signatures_season = RecipesEDAService.get_signatures_seasons(
            df_period, top_n=MAX_TOP_N)
        if not signatures_season:
            st.error("Could not compute season signatures !")
            return
        formatted = format_time(start, time.time())
        st.session_state["signatures_season"] = signatures_season
        st.toast(
            f"Signatures_season computed ({formatted})",
            icon=":material/thumb_up:",
            duration=5)

    assert all(season in seasons_list for season in list(signatures_season[0].keys())), (
        "Signatures list of countries does not match the list of countries in the dataset")

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
    st.header("Signature Ingredients")

    # Country selection dropdown
    default_country_name = "france"
    default_index = 0
    if default_country_name in countries_list:
        default_index = countries_list.index(default_country_name)

    selected_country = st.sidebar.selectbox(
        "Select a country",
        options=countries_list,
        index=default_index,
        placeholder="Select a country...",
        label_visibility='hidden'
    )

    selected_season = st.sidebar.selectbox(
        "Select a season",
        options=seasons_list,
        index=None,
        placeholder="Select a season...",
        label_visibility='hidden'
    )

    tab1, tab2 = st.tabs(["Country Analysis", "Season Analysis"])

    with tab1:
        if selected_country:
            # Top 3
            top_3_terms = list(
                signatures_country[0][selected_country].keys())[:3]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🥇 Top Signature", top_3_terms[0].title() if len(
                    top_3_terms) > 0 else "N/A")
            with col2:
                st.metric("🥈 Second Signature", top_3_terms[1].title() if len(
                    top_3_terms) > 1 else "N/A")
            with col3:
                st.metric("🥉 Third Signature", top_3_terms[2].title() if len(
                    top_3_terms) > 2 else "N/A")

            st.divider()

            left, right = st.columns(2, gap="large")
            with left:
                display_signature(
                    signatures_country[0][selected_country], selected_country, top_n_to_display, country=True)
            with right:
                display_signatures_tfidf_vs_tf(
                    signatures_country[0], signatures_country[1], selected_country, top_n_to_display)

    with tab2:
        if selected_season:
            left, right = st.columns(2, gap="large")
            with left:
                display_signature(signatures_season[0][selected_season], selected_season,
                                  top_n_to_display, country=False, season=True)
            with right:
                display_seasonal_pie(df_period)


if __name__ == "__main__":
    app()
