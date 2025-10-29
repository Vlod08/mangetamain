# app/pages/recipes/country_season_page.py
from __future__ import annotations
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

from mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.core.recipes_eda import RecipesEDAService

MAX_TOP_N = 20


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


if __name__ == "__main__":
    app()
