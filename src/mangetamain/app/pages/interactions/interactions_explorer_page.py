# app/pages/interactions/interactions_explorer_page.py
from __future__ import annotations
from pathlib import Path
import io
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from app.app_utils.ui import use_global_ui
from core.interactions_eda import InteractionsEDAService


def app():
    use_global_ui(
        page_title="Mangetamain — Exploratory Data Analysis of Interactions",
        subtitle="Quality of reviews, exploration and table (filters).",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
    )

    sns.set_theme()

    # ---- Sidebar : source & filters ----
    with st.sidebar:
        st.header("⚙️ Data Source")
        uploaded_file = st.file_uploader("Upload CSV/Parquet (optional)", type=["csv","parquet"])
        if uploaded_file is None:
            st.info("Using default dataset (Food.com reviews). You can upload your own CSV/Parquet file.")
        else:
            st.success(f"File uploaded: {Path(uploaded_file.name).name}")
    
        st.header("⚙️ Artefact")
        if st.button("🧹 Regenerate Clean Artefact"):
            with st.spinner("Preprocessing…"):
                interactions_eda_svc = InteractionsEDAService(uploaded_file=uploaded_file)
            st.success(f"Artefact regenerated successfully.")
            st.rerun()  # Refresh to load new artifact
        st.header("🎛️ Filters")
        rating_range = st.slider("Rating", 1.0, 5.0, value=(1.0, 5.0), step=0.5)
        min_len = st.number_input("Min. review length", min_value=0, value=0, step=10)

    # =========================
    # KPIs header
    # =========================

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(interactions_eda_svc.ds.df):,}")
    c2.metric("Columns", interactions_eda_svc.ds.df.shape[1])

    # % NA on the variable the most pertinent for Reviews (rating)
    if "rating" in interactions_eda_svc.ds.df.columns:
        pct_na_rating = interactions_eda_svc.ds.df["rating"].isna().mean() * 100
        c3.metric("% rating NA", f"{pct_na_rating:.1f}%")
    else:
        c3.metric("% rating NA", "—")
    import pandas as pd
    dates_ok = ("date" in interactions_eda_svc.ds.df.columns) \
        and pd.to_datetime(interactions_eda_svc.ds.df["date"], errors="coerce").notna().any()
    c4.metric("Dates parsées", "✅" if dates_ok else "—")

    # Short overview
    with st.expander("👀 Overview / Info"):
            st.dataframe(interactions_eda_svc.ds.df.head(20))
            buf = io.StringIO()
            interactions_eda_svc.ds.df.info(buf=buf)
            st.text(buf.getvalue())

    tabs = st.tabs(["🧹 Quality", "📊 Exploration", "📄 Table"])

    # =========================
    # 🧹 Quality
    # =========================
    with tabs[0]:
        st.subheader("Schema")
        st.dataframe(interactions_eda_svc.ds.schema())

        st.subheader("NaN rates (top 20)")
        st.dataframe(interactions_eda_svc.na_rate().head(20))

        st.subheader("Duplicates")
        dups = interactions_eda_svc.duplicates()
        for key, val in dups.items():
            if key != "full":
                st.write(f"Duplicates on {key.split('_')} : **{val}**")
            else:
                st.write(f"Duplicates (all columns) : **{val}**")

        st.subheader("Descriptive Statistics & Cardinalities")
        c1, c2 = st.columns(2)
        # Descriptive stats
        with c1:
            st.dataframe(interactions_eda_svc.desc_numeric())
        # Cardinalities
        with c2:
            st.dataframe(interactions_eda_svc.cardinalities().head(30))


    # =========================
    # 📊 Exploration
    # =========================
    with tabs[1]:
        st.caption(f"{len(interactions_eda_svc.ds.df):,} raws (before filtering)")

        colA, colB = st.columns(2)
        with colA:
            h = interactions_eda_svc.hist_rating()
            if not h.empty:
                st.plotly_chart(px.bar(h, x="left", y="count", title="Ratings distribution"))
        with colB:
            h2 = interactions_eda_svc.hist_review_len()
            if not h2.empty:
                st.plotly_chart(px.bar(h2, x="left", y="count", title="Review length (characters)"))

        bm = interactions_eda_svc.by_month()
        if not bm.empty:
            st.subheader("Trend over time")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.line(bm, x="month", y="n", title="Reviews per month"))
            with c2:
                st.plotly_chart(px.line(bm, x="month", y="mean_rating", title="Average rating per month"))

            yr = interactions_eda_svc.year_range()
            if yr:
                y = st.slider("Year (zoom)", yr[0], yr[1], value=int((yr[0]+yr[1])//2))
                oy = interactions_eda_svc.one_year(y)
                c3, c4 = st.columns(2)
                with c3:
                    fig, ax = plt.subplots(figsize=(9, 3.8))
                    ax.plot(oy["month"], oy["n"], marker="o")
                    ax.set_title(f"Volume per month — {y}")
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
        au = interactions_eda_svc.agg_by_user()
        if not au.empty:
            st.write("Top users (by #reviews):")
            st.dataframe(au.head(10))
        ar = interactions_eda_svc.agg_by_recipe()
        if not ar.empty:
            st.write("Top recipes (by #reviews):")
            st.dataframe(ar.head(10))

    # =========================
    # 📄 Table (with filters)
    # =========================
    with tabs[2]:
        year_opt = None
        yr = interactions_eda_svc.year_range()
        if yr:
            # nullable slider: we leave None by default with a checkbox if you prefer
            year_opt = st.slider("Year filter (optional)", yr[0], yr[1], value=None)

        fdf = interactions_eda_svc.apply_filters(rating_range=rating_range, min_len=min_len, year=year_opt)
        st.caption(f"{len(fdf):,} rows after filters")

        cols = [c for c in ["user_id", "recipe_id", "date", "rating", "review"] if c in fdf.columns]
        st.dataframe(fdf.head(1000)[cols], hide_index=True)

        st.download_button(
            "⬇️ Export CSV (filters)",
            fdf.to_csv(index=False).encode("utf-8"),
            "reviews_filtered.csv",
            "text/csv",
        )

if __name__ == "__main__":
    app()
