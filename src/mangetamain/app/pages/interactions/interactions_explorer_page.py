# app/pages/interactions/interactions_explorer_page.py
from __future__ import annotations
from pathlib import Path
import io
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.core.interactions_eda import InteractionsEDAService
from mangetamain.core.dataset import DatasetLoader


def app():
    use_global_ui(
        page_title="Mangetamain — Exploratory Data Analysis of Interactions",
        subtitle="Quality of reviews, exploration and table (filters).",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
    )

    sns.set_theme()

    # ======== Data Loading =========
    # Interactions dataset already uploaded in app entrypoint (main.py)
    interactions_df = st.session_state["interactions"]
    interactions_eda_svc = InteractionsEDAService()
    interactions_eda_svc.load(interactions_df, preprocess=False)

    interactions_columns = interactions_df.columns.values

    year_range = interactions_eda_svc.year_range()
    min_year, max_year = year_range if year_range else (2000, 2024)

    rating_range = interactions_eda_svc.rating_range()
    min_rating, max_rating = rating_range if rating_range else (1.0, 5.0)

    review_len_range = interactions_eda_svc.review_len_range()
    min_len, max_len = review_len_range if review_len_range else (0, 1000)

    # ---- Sidebar : source & filters ----
    with st.sidebar:
        # st.header("⚙️ Data Source")
        # uploaded_file = st.file_uploader("Upload CSV/Parquet (optional)", type=["csv","parquet"])
        # if uploaded_file is None:
        #     st.info("Using default dataset (Food.com reviews). You can upload your own CSV/Parquet file.")
        # else:
        #     st.success(f"File uploaded: {Path(uploaded_file.name).name}")
    
        # st.header("⚙️ Artefact")
        # if st.button("🧹 Regenerate Clean Artefact"):
        #     with st.spinner("Preprocessing…"):
        #         interactions_eda_svc = InteractionsEDAService(uploaded_file=uploaded_file)
        #     st.success(f"Artefact regenerated successfully.")
        #     st.rerun()  # Refresh to load new artifact
        st.header("Filters")
        rating_range = st.slider("Rating", min_rating, max_rating, value=(min_rating, max_rating), step=0.5)
        review_len_range = st.slider("Review length", min_len, max_len, value=(min_len, max_len))
        year_range = st.slider("Year range", min_year, max_year, value=(min_year, max_year))

    df_filtered = interactions_eda_svc.apply_filters(
        rating_range=rating_range,
        review_len_range=review_len_range,
        year_range=year_range,
    )

    interactions_eda_svc.load(df_filtered, preprocess=False)

    # =========================
    # KPIs header
    # =========================

    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df_filtered):,}")
    c2.metric("Columns", df_filtered.shape[1])

    # Short overview
    with st.expander("👀 Preview"):
        st.dataframe(df_filtered.head(20))
        # buf = io.StringIO()
        # df_filtered.info(buf=buf)
        # st.text(buf.getvalue())

    tabs = st.tabs(["🧹 Quality", "📊 Exploration", "📄 Table"])

    # =========================
    # 🧹 Quality
    # =========================
    if "issues" in st.session_state and "nan" in interactions_eda_svc.ds.issues:
        nan_dict = interactions_eda_svc.ds.issues["nan"]
    elif "nan" in interactions_eda_svc.ds.issues:
        nan_dict = interactions_eda_svc.ds.issues["nan"]
    else:
        nan_dict = {}

    if nan_dict:
        row = st.container(horizontal=True)
        with row:
            for col, na_val in nan_dict.items():
                if na_val > 0:
                    st.metric(f"{col} NA", na_val)

    with tabs[0]:
        st.subheader("Schema")
        st.dataframe(DatasetLoader.compute_schema(df_filtered))

        st.subheader("NaN rates")
        miss = interactions_eda_svc.na_counts()
        if miss.empty:
            st.write("No missing values detected in the dataset.")
        else:
            st.dataframe(miss)
            st.bar_chart(miss)

        st.subheader("Duplicates")
        dups = interactions_eda_svc.duplicates()
        if not dups:
            st.write("No duplicates found in the dataset.")
        else:
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
            st.dataframe(interactions_eda_svc.cardinalities())


    # =========================
    # 📊 Exploration
    # =========================
    with tabs[1]:
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

        st.subheader("Aggregations")
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
        cols = [c for c in ["user_id", "recipe_id", "date", "rating", "review"] if c in df_filtered.columns]
        st.dataframe(df_filtered.head(1000)[cols], hide_index=True)

        st.download_button(
            "⬇️ Export CSV (filters)",
            df_filtered.to_csv(index=False).encode("utf-8"),
            "reviews_filtered.csv",
            "text/csv",
        )

if __name__ == "__main__":
    app()
