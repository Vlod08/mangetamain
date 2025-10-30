# app/pages/02_Interactions_Advanced.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
import plotly.express as px
import seaborn as sns

from app.app_utils.ui import use_global_ui
from core.interactions_eda import InteractionsEDAService

use_global_ui(
    page_title="Mangetamain — Advanced Interactions Analysis",
    subtitle="Temporal trends, user bias, and text ↔ rating insights",
    logo="assets/mangetamain-logo.jpg",
    logo_size_px=90,
    round_logo=True,
)

sns.set_theme()

# we work on the clean parquet / dataset, no uploader here
svc = InteractionsEDAService(anchor=Path(__file__))
df = svc.load()

# ---------- KPI BAR ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(df):,}".replace(",", " "))
if "date" in df and df["date"].notna().any():
    c2.metric("Period", f"{df['date'].min().date()} → {df['date'].max().date()}")
else:
    c2.metric("Period", "—")
c3.metric("Unique users", f"{df['user_id'].nunique():,}" if "user_id" in df else "—")
c4.metric("Unique recipes", f"{df['recipe_id'].nunique():,}" if "recipe_id" in df else "—")

tabs = st.tabs(
    [
        "📆 Time series",
        "👥 User bias",
        "📝 Text ↔ Rating",
    ]
)

# =========================================================
# 1) TIME SERIES
# =========================================================
with tabs[0]:
    st.subheader("Monthly evolution (raw)")
    bm = svc.monthly_series()
    if bm.empty:
        st.info("No usable `date` column found.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.line(bm, x="month", y="n", title="Reviews per month"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                px.line(bm, x="month", y="mean_rating", title="Average rating per month"),
                use_container_width=True,
            )

        st.subheader("Smoothing (3-month rolling)")
        roll = svc.monthly_rolling(window=3)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                px.line(roll, x="month", y="n_roll3", title="Count (rolling 3)"),
                use_container_width=True,
            )
        with c4:
            st.plotly_chart(
                px.line(
                    roll,
                    x="month",
                    y="mean_rating_roll3",
                    title="Rating (rolling 3)",
                ),
                use_container_width=True,
            )

        st.subheader("Year-over-Year growth")
        yoy = svc.monthly_yoy()
        if not yoy.empty and "n_yoy" in yoy:
            fig = px.line(yoy, x="month", y="n_yoy", title="YoY on monthly volume")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Seasonal decomposition")
        dec = svc.seasonal_decompose_monthly()
        if dec.empty:
            st.info("Install `statsmodels` or provide more data to see the decomposition.")
        else:
            st.plotly_chart(
                px.line(
                    dec,
                    x="month",
                    y="value",
                    color="part",
                    title="Trend / seasonal / residual",
                ),
                use_container_width=True,
            )

        st.subheader("Anomalies (z-score)")
        an = svc.monthly_anomalies()
        if not an.empty:
            st.plotly_chart(
                px.scatter(
                    an,
                    x="month",
                    y="n",
                    color="is_anomaly",
                    hover_data=["z_n"],
                    title="Monthly volume anomalies",
                ),
                use_container_width=True,
            )

        st.subheader("Weekly / hourly profile")
        wk = svc.weekday_profile()
        if not wk.empty:
            wk["weekday"] = wk["wk"].map(
                {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            )
            st.plotly_chart(
                px.bar(wk, x="weekday", y="n", title="Reviews per weekday"),
                use_container_width=True,
            )
        heat = svc.weekday_hour_heat()
        if not heat.empty:
            st.plotly_chart(
                px.density_heatmap(
                    heat,
                    x="h",
                    y="wk",
                    z="n",
                    nbinsx=24,
                    nbinsy=7,
                    title="Hour × weekday heatmap",
                ),
                use_container_width=True,
            )

        st.subheader("User cohorts (quick view)")
        coh = svc.cohorts_users()
        if not coh.empty:
            # keep first 12 months of age
            small = coh[(coh["age"] >= 0) & (coh["age"] <= 12)]
            mat = small.pivot(index="cohort", columns="age", values="n").fillna(0)
            st.plotly_chart(
                px.imshow(
                    mat,
                    aspect="auto",
                    title="Cohorts (0 → 12 months from 1st review)",
                ),
                use_container_width=True,
            )

# =========================================================
# 2) USER BIAS
# =========================================================
with tabs[1]:
    st.subheader("User-level rating behaviour")
    ub = svc.user_bias()
    if ub.empty:
        st.info("Columns `user_id` / `rating` are missing.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(ub.head(20), use_container_width=True)
        with c2:
            st.plotly_chart(
                px.histogram(
                    ub,
                    x="mean",
                    nbins=25,
                    title="Distribution of user mean ratings",
                ),
                use_container_width=True,
            )

        st.plotly_chart(
            px.scatter(
                ub,
                x="n",
                y="mean",
                hover_data=["median"],
                title="#reviews vs average rating (by user)",
            ),
            use_container_width=True,
        )

    st.subheader("Rating vs review length")
    rvl = svc.rating_vs_length()
    if not rvl.empty:
        # trendline="ols" needs statsmodels, so we guard it
        try:
            fig = px.scatter(
                rvl,
                x="review_len",
                y="rating",
                opacity=0.2,
                trendline="ols",
                title="Rating vs review length",
            )
        except Exception:
            fig = px.scatter(
                rvl,
                x="review_len",
                y="rating",
                opacity=0.2,
                title="Rating vs review length",
            )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 3) TEXT ↔ RATING
# =========================================================
with tabs[2]:
    st.subheader("Most frequent tokens per rating bucket")
    tbr = svc.tokens_by_rating(k=20)
    if tbr.empty:
        st.info("Columns `review` / `rating` are missing.")
    else:
        st.dataframe(tbr.head(100), use_container_width=True)
        st.plotly_chart(
            px.bar(
                tbr,
                x="token",
                y="count",
                color="rating",
                barmode="group",
                title="Top tokens by (rounded) rating",
            ),
            use_container_width=True,
        )
