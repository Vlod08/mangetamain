import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from app.app_utils.io import load_data
from app.app_utils.filters import parse_tag_str
from app.app_utils.ui import use_global_ui
use_global_ui("Mangetamain — Événements & Saisons",     logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True, subtitle=None, wide=True)


from recipes_analyzer import RecipesAnalyzer

#st.title("📅 Événements & Saisons")

df = load_data().copy()
if "submitted" in df.columns and df["submitted"].notna().any():
    df["month"] = df["submitted"].dt.to_period("M").astype(str)
    st.bar_chart(df["month"].value_counts().sort_index())
else:
    st.info("Colonne 'submitted' indisponible.")

def season_from_month(m: int) -> str:
    # Hémisphère nord simple
    return ("Winter","Winter","Spring","Spring","Spring","Summer","Summer","Summer","Autumn","Autumn","Autumn","Winter")[m-1]

if df["submitted"].notna().any():
    df["season"] = df["submitted"].dt.month.apply(season_from_month)
    season = st.selectbox("Saison", sorted(df["season"].dropna().unique()))
    sub = df[df["season"]==season]
else:
    season = "All"; sub = df

events = ["christmas","easter","bbq","thanksgiving","halloween"]
event = st.selectbox("Événement (tags)", ["(aucun)"] + events)
if event != "(aucun)":
    sub = sub[sub["tags"].apply(lambda x: event in [t.lower() for t in parse_tag_str(x)])]

st.subheader("Suggestions")
cols = st.columns(3)
for i, row in sub.head(21).iterrows():
    with cols[i%3]:
        st.markdown(f"**{row['name']}**")
        st.caption(f"{int(row['minutes']) if pd.notna(row['minutes']) else '—'} min • {row['n_steps']} steps")



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
def get_all_season_signatures(_analyzer: RecipesAnalyzer) -> dict: # <-- top_n removed from arguments
    """
    Executes the costly TF-IDF calculation for seasons (Top 20 max) 
    and caches the result.
    """
    MAX_TOP_N = 20 # Always calculate the maximum possible for the slider
    st.write(f"Calculating TF-IDF for seasons (Top {MAX_TOP_N})... (runs only once or if _analyzer changes)")   
    df_processed = _analyzer.recipes_df 
    # Call the method requesting the Top 20
    signatures = _analyzer.get_signatures_seasons(df_processed, MAX_TOP_N) 
    # Corrected variable name from signatures_top20 to signatures
    st.success(f"Signatures (Top {MAX_TOP_N}) calculated for {len(signatures)} seasons.") 
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

# --- Main Section: season Signatures ---
st.header("Signature Ingredients by Season")

# Fetch the cached dictionary (assumed to contain Top 20 sorted scores per season)
# 'analyzer' must be defined and loaded earlier using @st.cache_resource
# 'get_all_season_signatures' must be defined earlier using @st.cache_data
all_signatures = get_all_season_signatures(analyzer)

if all_signatures:
    # Get season list for the dropdown
    seasons_list = sorted(list(all_signatures.keys()))

    # season selection dropdown
    selected_season = st.selectbox(
        "Choose a season:",
        options=seasons_list,
        index=None,
        placeholder="Select a season..."
    )

    # Display results if a season is selected
    if selected_season:
        # Get the sorted score dictionary for the selected season (up to Top 20)
        scores_dict_top20_sorted = all_signatures[selected_season]

        if scores_dict_top20_sorted:

            # --- FILTER TOP N BASED ON SLIDER ---
            # Since the dict is sorted, take the first N items based on the slider value
            # Convert items to list, slice, convert back to dict
            scores_to_display = dict(list(scores_dict_top20_sorted.items())[:top_n_to_display])
            # ------------------------------------

            st.subheader(f"Word Cloud for Top {top_n_to_display} Ingredients in: {selected_season}")

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
            st.warning(f"No signature scores found for '{selected_season}'.")
else:
    st.error("Could not compute season signatures. Check the 'get_all_season_signatures' function.")