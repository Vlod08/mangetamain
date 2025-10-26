# app/pages/01_Explorer_Reviews.py
from __future__ import annotations
from pathlib import Path
import sys
import io
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from app.app_utils.ui import use_global_ui
from core.recipes_preprocessing import RecipesPreprocessor
from core.service_reviews import ReviewsEDAService

use_global_ui(
    page_title="Mangetamain — Analyse exploratoires des données des Interactions",
    subtitle="Qualité des avis, exploration et table (filtres).",
    logo="image/image.jpg",
    logo_size_px=90,
    round_logo=True,
)

sns.set_theme()

# ---- Sidebar : source & filtres ----
with st.sidebar:
    st.header("⚙️ Données")
    uploaded = st.file_uploader("Upload CSV/Parquet (optionnel)", type=["csv","parquet"])
with st.sidebar:
    st.header("⚙️ Artefact")
    if st.button("🧹 Régénérer l’artefact propre"):
        with st.spinner("Prétraitement…"):
            out = RecipesPreprocessor(anchor=Path(__file__)).run()
        st.success(f"Artefact régénéré : {out}. Recharge la page (Ctrl/Cmd+R).")
    st.header("🎛️ Filtres")
    rating_range = st.slider("Rating", 1.0, 5.0, value=(1.0, 5.0), step=0.5)
    min_len = st.number_input("Longueur min. review", min_value=0, value=0, step=10)



# (optionnel) pipeline RAW -> CLEAN comme Recipes
#with st.expander("Pipeline RAW → CLEAN (artefact Reviews)", expanded=False):
    #st.caption("Exécute le script de prétraitement pour générer/mettre à jour l'artefact reviews_clean.parquet.")
    #if st.button("🧹 Régénérer l’artefact Reviews"):
        #import os
        #code = os.system("poetry run python src/core/interactions_preprocess.py")
        #st.success("Artefact Reviews régénéré. Recharge la page (Ctrl/Cmd+R).") if code == 0 else st.error("Échec génération.")

svc = ReviewsEDAService(anchor=Path(__file__), uploaded_file=uploaded)
df = svc.load()
# --- KPIs header ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{len(df):,}")
c2.metric("Colonnes", df.shape[1])

# % NA sur la variable la + pertinente pour Reviews (rating)
if "rating" in df.columns:
    pct_na_rating = df["rating"].isna().mean() * 100
    c3.metric("% rating NA", f"{pct_na_rating:.1f}%")
else:
    c3.metric("% rating NA", "—")
import pandas as pd
dates_ok = ("date" in df.columns) and pd.to_datetime(df["date"], errors="coerce").notna().any()
c4.metric("Dates parsées", "✅" if dates_ok else "—")
    # petit aperçu
with st.expander("👀 Aperçu / info"):
        st.dataframe(df.head(20), width="stretch")
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())
#st.tabs(["Qualité", "Exploration", "Table"])

#tab_quality, tab_explo, tab_table = st.tabs(["Qualité", "Exploration", "Table"])
tabs = st.tabs(["🧹 Qualité", "📊 Exploration", "📄 Table"])

# =========================
# 🧹 Qualité
# =========================
with tabs[0]:
    st.subheader("Schéma")
    st.dataframe(svc.schema(), width="stretch")

    st.subheader("Taux de NA (top)")
    st.dataframe(svc.na_rate().head(20), width="stretch")

    dups = svc.duplicates()
    st.write(f"Duplicats (toutes colonnes) : **{dups['dup_total']}**")
    if dups["dup_on_keys"] is not None:
        st.write(f"Duplicats sur {dups['keys']} : **{dups['dup_on_keys']}**")

    st.subheader("Numériques & cardinalités")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(svc.desc_numeric(), width="stretch")
    with c2:
        st.dataframe(svc.cardinalities().head(30), width="stretch")



# =========================
# 📊 Exploration
# =========================
with tabs[1]:
    st.caption(f"{len(df):,} lignes (avant filtres)")

    colA, colB = st.columns(2)
    with colA:
        h = svc.hist_rating()
        if not h.empty:
            st.plotly_chart(px.bar(h, x="left", y="count", title="Distribution des ratings"), width="stretch")
    with colB:
        h2 = svc.hist_review_len()
        if not h2.empty:
            st.plotly_chart(px.bar(h2, x="left", y="count", title="Longueur des reviews (caractères)"), width="stretch")

    bm = svc.by_month()
    if not bm.empty:
        st.subheader("Temporalité")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(bm, x="month", y="n", title="Reviews par mois"), width="stretch")
        with c2:
            st.plotly_chart(px.line(bm, x="month", y="mean_rating", title="Rating moyen par mois"), width="stretch")

        yr = svc.year_range()
        if yr:
            y = st.slider("Année (zoom)", yr[0], yr[1], value=int((yr[0]+yr[1])//2))
            oy = svc.one_year(y)
            c3, c4 = st.columns(2)
            with c3:
                fig, ax = plt.subplots(figsize=(9, 3.8))
                ax.plot(oy["month"], oy["n"], marker="o")
                ax.set_title(f"Volume par mois — {y}")
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                st.pyplot(fig, clear_figure=True)
            with c4:
                fig, ax = plt.subplots(figsize=(9, 3.8))
                ax.plot(oy["month"], oy["mean_rating"], marker="o")
                ax.set_title(f"Rating moyen par mois — {y}")
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                st.pyplot(fig, clear_figure=True)

    st.subheader("Agrégations")
    au = svc.agg_by_user()
    if not au.empty:
        st.write("Top utilisateurs (par #reviews) :")
        st.dataframe(au.head(10), width="stretch")
    ar = svc.agg_by_recipe()
    if not ar.empty:
        st.write("Top recettes (par #reviews) :")
        st.dataframe(ar.head(10), width="stretch")

# =========================
# 📄 Table (avec filtres)
# =========================
with tabs[2]:
    year_opt = None
    yr = svc.year_range()
    if yr:
        # slider nullable : on laisse None par défaut avec une case à cocher si tu préfères
        year_opt = st.slider("Filtre année (optionnel)", yr[0], yr[1], value=None)

    fdf = svc.apply_filters(rating_range=rating_range, min_len=min_len, year=year_opt)
    st.caption(f"{len(fdf):,} lignes après filtres")

    cols = [c for c in ["user_id", "recipe_id", "date", "rating", "review"] if c in fdf.columns]
    st.dataframe(fdf.head(1000)[cols], width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Export CSV (filtres)",
        fdf.to_csv(index=False).encode("utf-8"),
        "reviews_filtered.csv",
        "text/csv",
    )
