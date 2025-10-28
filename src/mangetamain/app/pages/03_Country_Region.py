from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from app_utils.io import load_data
from app_utils.country import add_country_column
from app_utils.viz import bar_top_counts
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain —  Pays & Régions",     logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True, subtitle=None, wide=True)

from recipes_analyzer import RecipesAnalyzer


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


# -----------------------------------------------------------------
# STEP 1: CACHED FUNCTIONS (THE "BRAIN")
# -----------------------------------------------------------------

@st.cache_resource
def load_analyzer():
    """
    Instantiates and initializes RecipesAnalyzer. 
    Includes loading and preprocessing data. Executed only once.
    """
    st.write("Initializing RecipesAnalyzer (runs only once)...")
    analyzer = RecipesAnalyzer()
    analyzer.analyze()
    st.success("RecipesAnalyzer initialized and base data analyzed.")
    return analyzer

@st.cache_data
def get_all_country_signatures(_analyzer: RecipesAnalyzer) -> dict: # <-- top_n removed from arguments
    """
    Executes the costly TF-IDF calculation for countries (Top 20 max) 
    and caches the result.
    """
    MAX_TOP_N = 20 # Always calculate the maximum possible for the slider
    st.write(f"Calculating TF-IDF for countries (Top {MAX_TOP_N})... (runs only once or if _analyzer changes)")   
    df_processed = _analyzer.recipes_df 
    # Call the method requesting the Top 20
    signatures = _analyzer.get_signatures_countries(df_processed, MAX_TOP_N) 
    # Corrected variable name from signatures_top20 to signatures
    st.success(f"Signatures (Top {MAX_TOP_N}) calculated for {len(signatures)} countries.") 
    return signatures

# -----------------------------------------------------------------
# ÉTAPE 2 : L'INTERFACE UTILISATEUR (L'AFFICHAGE)
# -----------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Analyseur de Recettes et Signatures d'Ingrédients 🧑‍🍳")

# --- Chargement initial (utilise le cache) ---
with st.spinner("Loading analyzer and initial data..."):
    analyzer = load_analyzer()

# --- Sidebar Controls ---
st.sidebar.header("Analysis Options")
# Slider to select how many of the pre-calculated top ingredients to display
top_n_to_display = st.sidebar.slider(
    "Display Top N Ingredients:",
    min_value=5,
    max_value=20, # Should match the MAX_TOP_N used in the cached function
    value=10      # Default value
)

# --- Main Section: Country Signatures ---
st.header("Signature Ingredients by Country")

# Fetch the cached dictionary (assumed to contain Top 20 sorted scores per country)
# 'analyzer' must be defined and loaded earlier using @st.cache_resource
# 'get_all_country_signatures' must be defined earlier using @st.cache_data
all_signatures = get_all_country_signatures(analyzer)

if all_signatures:
    # Get country list for the dropdown
    countries_list = sorted(list(all_signatures.keys()))

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
        scores_dict_top20_sorted = all_signatures[selected_country]

        if scores_dict_top20_sorted:

            # --- FILTER TOP N BASED ON SLIDER ---
            # Since the dict is sorted, take the first N items based on the slider value
            # Convert items to list, slice, convert back to dict
            scores_to_display = dict(list(scores_dict_top20_sorted.items())[:top_n_to_display])
            # ------------------------------------

            st.subheader(f"Word Cloud for Top {top_n_to_display} Ingredients in: {selected_country}")

            if scores_to_display:
                try:
                    # Generate Word Cloud from the filtered scores
                    wc = WordCloud(width=800,
                                   height=400,
                                   background_color='white',
                                   colormap='viridis' # Other options: 'plasma', 'magma'
                                  ).generate_from_frequencies(scores_to_display)

                    # Display Word Cloud using matplotlib
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error generating word cloud: {e}")
            else:
                 st.info("No ingredients to display for the selected Top N value.")

        else:
            st.warning(f"No signature scores found for '{selected_country}'.")
else:
    st.error("Could not compute country signatures. Check the 'get_all_country_signatures' function.")

