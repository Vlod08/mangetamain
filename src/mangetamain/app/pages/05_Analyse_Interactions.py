# app/pages/02_Reviews_Advanced.py
from __future__ import annotations
from pathlib import Path
import sys

import streamlit as st
import seaborn as sns
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app_utils.ui import use_global_ui
from core.service_reviews import ReviewsEDAService

use_global_ui(
    page_title="Mangetamain — Reviews : Analyse avancée",
    subtitle="Corrélations, biais utilisateurs, texte ↔ rating.",
    logo="image/image.jpg",
    logo_size_px=90,
    round_logo=True,
)

sns.set_theme()


svc = ReviewsEDAService(anchor=Path(__file__), uploaded_file=uploaded)

tabs = st.tabs(["📈 Corrélations", "👥 Biais utilisateurs", "📝 Texte ↔ Rating"])

# 1) Corrélations
with tabs[0]:
    corr = svc.corr_numeric()
    if corr.empty:
        st.info("Pas assez de colonnes numériques.")
    else:
        fig = px.imshow(
            corr,
            title="Heatmap des corrélations (numériques)",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig, width="stretch")
    rvl = svc.rating_vs_length()
    if not rvl.empty:
        st.plotly_chart(
            px.scatter(
                rvl,
                x="review_len",
                y="rating",
                opacity=0.2,
                trendline="ols",
                title="Rating vs longueur de review",
            ),
            width="stretch",
        )

# 2) Biais utilisateurs
with tabs[1]:
    ub = svc.user_bias()
    if ub.empty:
        st.info("Colonnes 'user_id'/'rating' absentes.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(ub.head(20), width="stretch")
        with col2:
            st.plotly_chart(
                px.histogram(
                    ub,
                    x="mean",
                    nbins=25,
                    title="Distribution des moyennes par utilisateur",
                ),
                width="stretch",
            )
        st.plotly_chart(
            px.scatter(
                ub,
                x="n",
                y="mean",
                title="Volume de reviews vs moyenne",
                hover_data=["median"],
            ),
            width="stretch",
        )

# 3) Texte ↔ Rating (top tokens par rating arrondi)
with tabs[2]:
    tbr = svc.tokens_by_rating(k=20)
    if tbr.empty:
        st.info("Colonnes 'review'/'rating' absentes.")
    else:
        st.dataframe(tbr.head(100), width="stretch")
        st.plotly_chart(
            px.bar(
                tbr,
                x="token",
                y="count",
                color="rating",
                barmode="group",
                title="Top tokens par rating (arrondi)",
            ),
            width="stretch",
        )
