# app/pages/interactions/interactions_analysis_page.py
from __future__ import annotations
import streamlit as st
import seaborn as sns
import plotly.express as px

from app.app_utils.ui import use_global_ui
from core.interactions_eda import InteractionsEDAService


def app():
    use_global_ui(
        page_title="Mangetamain — Interactions: Advanced Analysis",
        subtitle="Correlation, user bias, text ↔ rating.",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
    )

    sns.set_theme()

    svc = InteractionsEDAService()

    tabs = st.tabs(["📈 Correlations", "👥 User Bias", "📝 Text ↔ Rating"])

    # 1) Correlations
    with tabs[0]:
        corr = svc.corr_numeric()
        if corr.empty:
            st.info("Not enough numeric columns.")
        else:
            fig = px.imshow(
                corr,
                title="Heatmap of Correlations (Numeric)",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            st.plotly_chart(fig)
        rvl = svc.rating_vs_length()
        if not rvl.empty:
            st.plotly_chart(
                px.scatter(
                    rvl,
                    x="review_len",
                    y="rating",
                    opacity=0.2,
                    trendline="ols",
                    title="Rating vs Review Length",
                )
            )

    # 2) User Bias
    with tabs[1]:
        ub = svc.user_bias()
        if ub.empty:
            st.info("Columns 'user_id'/'rating' missing.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(ub.head(20))
            with col2:
                st.plotly_chart(
                    px.histogram(
                        ub, x="mean", nbins=25, title="Distribution of Means by User"
                    ),
                    width="stretch",
                )
            st.plotly_chart(
                px.scatter(
                    ub,
                    x="n",
                    y="mean",
                    title="Volume of Reviews vs Mean",
                    hover_data=["median"],
                ),
                width="stretch",
            )

    # 3) Text ↔ Rating (top tokens by rounded rating)
    with tabs[2]:
        tbr = svc.tokens_by_rating(k=20)
        if tbr.empty:
            st.info("Columns 'review'/'rating' missing.")
        else:
            st.dataframe(tbr.head(100))
            st.plotly_chart(
                px.bar(
                    tbr,
                    x="token",
                    y="count",
                    color="rating",
                    barmode="group",
                    title="Top tokens by Rounded Rating",
                )
            )


if __name__ == "__main__":
    app()
