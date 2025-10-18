import streamlit as st
from app_utils.io import load_data
from app_utils.filters import ensure_session_filters, apply_basic_filters
from app_utils.viz import scatter_time_steps, hist_minutes, hist_steps
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain —  Explorateur — Analyse interactive",     logo="image/image.jpg",
    subtitle="Analyse interactive des recettes avec filtres sur temps, étapes, tags et ingrédients.",
    wide=True,
    logo_size_px=90,
    round_logo=True)

#st.title("🔎 Explorateur — Analyse interactive")
ensure_session_filters()

df = load_data()

with st.sidebar:
    st.header("Filtres")
    m_min, m_max = st.slider("Minutes", 0, int(df["minutes"].max(skipna=True) or 240), (0, 120))
    s_min, s_max = st.slider("Étapes", 0, int(df["n_steps"].max(skipna=True) or 20), (0, 12))
    inc_tags = st.text_input("Inclure tags (séparés par ,)", "")
    exc_tags = st.text_input("Exclure tags (séparés par ,)", "")
    inc_ings = st.text_input("Contient ingrédients (séparés par ,)", "")
    st.session_state["filters"].update({
        "minutes": (m_min, m_max),
        "steps": (s_min, s_max),
        "include_tags": [t.strip() for t in inc_tags.split(",") if t.strip()],
        "exclude_tags": [t.strip() for t in exc_tags.split(",") if t.strip()],
        "include_ings": [t.strip() for t in inc_ings.split(",") if t.strip()],
    })

fdf = apply_basic_filters(df)
st.caption(f"{len(fdf):,} recettes après filtres")

tab1, tab2, tab3 = st.tabs(["Temps & Effort", "Distributions", "Table"])
with tab1:
    st.plotly_chart(scatter_time_steps(fdf), use_container_width=True)
with tab2:
    c1, c2 = st.columns(2)
    c1.plotly_chart(hist_minutes(fdf), use_container_width=True)
    c2.plotly_chart(hist_steps(fdf), use_container_width=True)
with tab3:
    st.dataframe(fdf[["name","minutes","n_steps","n_ingredients","tags"]].head(1000),
                 use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export CSV (filtres)", fdf.to_csv(index=False).encode("utf-8"),
                       "recipes_filtered.csv", "text/csv")
