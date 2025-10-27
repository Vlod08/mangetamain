# streamlit run app.py
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from map import BubbleMapFolium

st.set_page_config(page_title="Recipes Bubble Map", layout="wide")
st.title("Recipes Bubble Map")

# Data source selection
file = r"..\..\data\raw_data\recipes_raw_sample.csv" # to be replaced by the right path #TODO
if not file:
    st.stop()
df_raw = pd.read_csv(file)

# Single control mode: country or continent
level = st.radio("View", ["country", "continent"], horizontal=True)

# Build aggregated counts for the chosen control mode
mapper = BubbleMapFolium(tiles="OpenStreetMap", auto_centroids=True)

if level == "country":
    counts = mapper.counts_by_country(df_raw, country_col="country")       # -> ['country','count_recipes']
else:
    counts = mapper.counts_by_continent(df_raw, continent_col="continent") # -> ['continent','count_recipes']

# Optional selection: highlight & color the country's surface (country view only)
selected_country = None
if level == "country":
    country_list = counts["country"].dropna().astype(str).sort_values().unique().tolist()
    choice = st.selectbox("Choisir un pays à surligner", ["(aucun)"] + country_list, index=0)
    selected_country = None if choice == "(aucun)" else choice

# Build the map (segmented bubble colors + soft polygon fill for selected country)
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
    zoom_mode="continent",
)

# Display the map
result= st_folium(m, height=640, width=None)