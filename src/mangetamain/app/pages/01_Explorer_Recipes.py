# src/app/pages/01_Data_Explorer.py
from __future__ import annotations
from pathlib import Path
import sys
import io
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app_utils.ui import use_global_ui
from core.recipes_service import RecipesEDAService
from core.recipes_preprocessing import RecipesPreprocessor

use_global_ui(
    page_title="Mangetamain — Analyse exploratoires des données Recettes",
    subtitle="Vérification du schéma, qualité des données, filtres et visualisations.",
    logo="image/image.jpg",
    logo_size_px=90,
    round_logo=True,
    wide=True,
)

svc = RecipesEDAService(anchor=Path(__file__))
df = svc.load()

# --------- Barre latérale : artefact + filtres ----------
with st.sidebar:
    st.header("⚙️ Données")
    uploaded = st.file_uploader(
        "Upload CSV/Parquet (optionnel)", type=["csv", "parquet"]
    )

with st.sidebar:
    st.header("⚙️ Artefact")
    if st.button("🧹 Régénérer l’artefact propre"):
        with st.spinner("Prétraitement…"):
            out = RecipesPreprocessor(anchor=Path(__file__)).run()
        st.success(f"Artefact régénéré : {out}. Recharge la page (Ctrl/Cmd+R).")

    st.header("Filtres")
    m_lo, m_hi = 0, int(df["minutes"].max(skipna=True) or 240)
    s_lo, s_hi = 0, int(df["n_steps"].max(skipna=True) or 20)
    minutes = st.slider("Minutes", m_lo, m_hi, (0, min(120, m_hi)))
    steps = st.slider("Étapes", s_lo, s_hi, (0, min(12, s_hi)))
    inc_tags = [
        t.strip()
        for t in st.text_input("Inclure tags ( , )", "").split(",")
        if t.strip()
    ]
    exc_tags = [
        t.strip()
        for t in st.text_input("Exclure tags ( , )", "").split(",")
        if t.strip()
    ]
    inc_ings = [
        t.strip()
        for t in st.text_input("Contient ingrédients ( , )", "").split(",")
        if t.strip()
    ]

# --------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{len(df):,}")
c2.metric("Colonnes", df.shape[1])
c3.metric(
    "% minutes NA",
    f"{(df['minutes'].isna().mean()*100):.1f}%" if "minutes" in df else "—",
)
c4.metric(
    "Dates parsées",
    "✅" if "submitted" in df and df["submitted"].notna().any() else "—",
)

# --------- Onglets ----------
tab1, tab2, tab3 = st.tabs(["🧹 Qualité", "📊 Exploration", "📄 Table"])

# ---- Qualité ----
with tab1:
    st.subheader("Schéma & complétude")
    with st.expander("👀 Aperçu / schéma"):
        st.dataframe(df.head(20), use_container_width=True)
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())
        st.dataframe(svc.schema(), use_container_width=True)

    st.subheader("Manquants (Top 10)")
    miss = svc.na_rate().head(10) * 100
    st.bar_chart(miss)

    dup = svc.duplicates()
    st.write(f"Dupliqués (toutes colonnes) : **{dup['dup_total']}**")
    if dup["dup_on_keys"] is not None:
        st.write(f"Dupliqués sur {dup['keys']} : **{dup['dup_on_keys']}**")

    st.subheader("Numériques & cardinalités")
    st.dataframe(svc.numeric_desc().head(20), use_container_width=True)
    st.dataframe(svc.cardinalities().head(30), use_container_width=True)

# ---- Exploration ----
with tab2:
    fdf = svc.apply_filters(
        minutes=minutes,
        steps=steps,
        include_tags=inc_tags,
        exclude_tags=exc_tags,
        include_ings=inc_ings,
    )
    st.caption(f"{len(fdf):,} recettes après filtres")

    c1, c2 = st.columns(2)
    hmin = svc.minutes_hist()
    if not hmin.empty:
        c1.plotly_chart(
            px.bar(hmin, x="left", y="count", title="Distribution des minutes"),
            use_container_width=True,
        )
    hstp = svc.steps_hist()
    if not hstp.empty:
        c2.plotly_chart(
            px.bar(hstp, x="left", y="count", title="Distribution des étapes"),
            use_container_width=True,
        )

    byy = svc.by_year()
    if not byy.empty:
        st.plotly_chart(
            px.line(byy, x="year", y="n", title="Recettes par année"),
            use_container_width=True,
        )

    top_ing = svc.top_ingredients(30)
    if not top_ing.empty:
        st.subheader("Top ingrédients")
        st.plotly_chart(
            px.bar(top_ing.head(20), x="ingredient", y="count"),
            use_container_width=True,
        )

# ---- Table ----
with tab3:
    cols = [
        c
        for c in ["name", "minutes", "n_steps", "n_ingredients", "tags"]
        if c in df.columns
    ]
    view = svc.apply_filters(
        minutes=minutes,
        steps=steps,
        include_tags=inc_tags,
        exclude_tags=exc_tags,
        include_ings=inc_ings,
    )
    st.dataframe(view[cols].head(1000), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Export CSV (filtres)",
        view.to_csv(index=False).encode("utf-8"),
        "recipes_filtered.csv",
        "text/csv",
    )
