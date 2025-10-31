# app/pages/interactions/interactions_explorer_page.py
from __future__ import annotations
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
    interactions_df = st.session_state["interactions"]  # already loaded in main.py
    interactions_eda_svc = InteractionsEDAService()
    interactions_eda_svc.load(interactions_df, preprocess=False)

    # Ranges for sliders
    year_range = interactions_eda_svc.year_range()
    min_year, max_year = year_range if year_range else (2000, 2024)

    rating_range = interactions_eda_svc.rating_range()
    min_rating, max_rating = rating_range if rating_range else (1.0, 5.0)

    review_len_range = interactions_eda_svc.review_len_range()
    min_len, max_len = review_len_range if review_len_range else (0, 1000)

    # ---- Sidebar filters ----
    with st.sidebar:
        st.header("Filters")
        rating_range = st.slider(
            "Rating", min_rating, max_rating, value=(min_rating, max_rating), step=0.5
        )
        review_len_range = st.slider(
            "Review length", min_len, max_len, value=(min_len, max_len)
        )
        year_range = st.slider(
            "Year range", min_year, max_year, value=(min_year, max_year)
        )

    df_filtered = interactions_eda_svc.apply_filters(
        rating_range=rating_range,
        review_len_range=review_len_range,
        year_range=year_range,
    )

    # Reload the service with filtered data so downstream methods reflect filters
    interactions_eda_svc.load(df_filtered, preprocess=False)

    # =========================
    # KPIs header
    # =========================
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df_filtered):,}")
    c2.metric("Columns", df_filtered.shape[1])

    # Preview
    with st.expander("👀 Preview"):
        st.dataframe(df_filtered.head(20), width="stretch")

    tabs = st.tabs(["🧹 Quality", "📊 Exploration", "📄 Table"])

    # =========================
    # 🧹 Quality (all based on df_filtered)
    # =========================
    with tabs[0]:
        st.subheader("Schema")
        st.dataframe(DatasetLoader.compute_schema(df_filtered), width="stretch")

        st.subheader("NaN overview (counters)")
        nan_counts = df_filtered.isna().sum().sort_values(ascending=False)
        nan_counts = nan_counts[nan_counts > 0]
        if nan_counts.empty:
            st.write("No missing values detected in the filtered dataset.")
        else:
            cols = st.columns(min(4, len(nan_counts)))
            for i, (colname, count) in enumerate(nan_counts.items()):
                with cols[i % len(cols)]:
                    st.metric(f"{colname} NA", int(count))

        st.subheader("NaN rates (table + bar)")
        if not nan_counts.empty:
            miss_df = (
                nan_counts.to_frame("n_na")
                .assign(rate=lambda d: d["n_na"] / len(df_filtered))
                .reset_index()
                .rename(columns={"index": "column"})
            )
            st.dataframe(miss_df, width="stretch")
            st.bar_chart(miss_df.set_index("column")["n_na"])
        else:
            st.write("—")

        st.subheader("Duplicates")
        dups = (
            interactions_eda_svc.duplicates()
        )  # computed on filtered df thanks to .load() above
        if not dups:
            st.write("No duplicates found in the filtered dataset.")
        else:
            for key, val in dups.items():
                label = " ".join(key.split("_")) if key != "full" else "all columns"
                st.write(f"Duplicates on {label}: **{val}**")

        st.subheader("Descriptive Statistics & Cardinalities")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(interactions_eda_svc.desc_numeric(), width="stretch")
        with c2:
            st.dataframe(interactions_eda_svc.cardinalities(), width="stretch")

    # =========================
    # 📊 Exploration
    # =========================
    with tabs[1]:
        colA, colB = st.columns(2)
        with colA:
            h = interactions_eda_svc.hist_rating()
            if not h.empty:
                st.plotly_chart(
                    px.bar(h, x="left", y="count", title="Ratings distribution"),
                    config={"width": "stretch"},
                )
        with colB:
            h2 = interactions_eda_svc.hist_review_len()
            if not h2.empty:
                st.plotly_chart(
                    px.bar(h2, x="left", y="count", title="Review length (characters)"),
                    config={"width": "stretch"},
                )

        bm = interactions_eda_svc.by_month()
        if not bm.empty:
            st.subheader("Trend over time")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.line(bm, x="month", y="n", title="Reviews per month"),
                    config={"width": "stretch"},
                )
            with c2:
                st.plotly_chart(
                    px.line(
                        bm, x="month", y="mean_rating", title="Average rating per month"
                    ),
                    config={"width": "stretch"},
                )

            yr = interactions_eda_svc.year_range()
            if yr:
                y = st.slider(
                    "Year (zoom)", yr[0], yr[1], value=int((yr[0] + yr[1]) // 2)
                )
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
                    ax.set_title(f"Average rating per month — {y}")
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                    st.pyplot(fig, clear_figure=True)

        st.subheader("Aggregations")
        au = interactions_eda_svc.agg_by_user()
        if not au.empty:
            st.write("Top users (by #reviews):")
            st.dataframe(au.head(10), width="stretch")
        ar = interactions_eda_svc.agg_by_recipe()
        if not ar.empty:
            st.write("Top recipes (by #reviews):")
            st.dataframe(ar.head(10), width="stretch")

    # =========================
    # 📄 Table (with filters)
    # =========================
    with tabs[2]:
        cols = [
            c
            for c in ["user_id", "recipe_id", "date", "rating", "review"]
            if c in df_filtered.columns
        ]
        st.dataframe(df_filtered.head(1000)[cols], hide_index=True, width="stretch")

        st.download_button(
            "⬇️ Export CSV (filters)",
            df_filtered.to_csv(index=False).encode("utf-8"),
            "reviews_filtered.csv",
            "text/csv",
        )


if __name__ == "__main__":
    app()
