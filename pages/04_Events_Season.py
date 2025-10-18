import streamlit as st
import pandas as pd
from app_utils.io import load_data
from app_utils.filters import parse_tag_str
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain — Événements & Saisons",     logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True, subtitle=None, wide=True)


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
