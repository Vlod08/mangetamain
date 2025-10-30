# app/pages/interactions/interactions_explorer_page.py
from __future__ import annotations
from pathlib import Path
import io
import pandas as pd

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

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("⚙️ Data source")
        uploaded_file = st.file_uploader(
            "Upload CSV/Parquet (optional)", type=["csv", "parquet"]
        )

        st.header("⚙️ Artefact")
        if st.button("🧹 Regenerate clean artefact"):
            st.success("Artefact regenerated (placeholder).")
            st.rerun()

        st.header("🎛️ Filters")
        rating_range = st.slider(
            "Rating", 1.0, 5.0, value=(1.0, 5.0), step=0.5, key="filter_rating"
        )
        min_len = st.number_input(
            "Min. review length", min_value=0, value=0, step=10, key="filter_min_len"
        )

    # ---------- SERVICE + DATA ----------
    svc = InteractionsEDAService(
        anchor=Path(__file__),
        uploaded_file=uploaded_file,
    )
    df = svc.load()

    # ---------- KPI HEADER ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])

    if "rating" in df.columns:
        pct_na_rating = df["rating"].isna().mean() * 100
        c3.metric("% rating NA", f"{pct_na_rating:.1f}%")
    else:
        c3.metric("% rating NA", "—")

    dates_ok = (
        "date" in df.columns
        and pd.to_datetime(df["date"], errors="coerce").notna().any()
    )
    c4.metric("Parsed dates", "✅" if dates_ok else "—")

    # ---------- OVERVIEW ----------
    with st.expander("👀 Overview / Info"):
        st.dataframe(df.head(20), use_container_width=True)
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())

    tabs = st.tabs(["🧹 Quality", "📊 Exploration", "📄 Table"])

    # ==================== 1) QUALITY ====================
    with tabs[0]:
        st.subheader("Schema")
        st.dataframe(
            pd.DataFrame({"col": df.columns, "dtype": df.dtypes.astype(str)})
        )

        st.subheader("NaN rates (top 20)")
        st.dataframe(svc.na_rate().head(20))

        st.subheader("Duplicates")
        dups = svc.duplicates()
        if dups:
            for k, v in dups.items():
                st.write(f"{k}: **{v}**")
        else:
            st.write("No duplicates detected on key columns.")

        st.subheader("Descriptive statistics & cardinalities")
        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(svc.desc_numeric(), use_container_width=True)
        with col_b:
            st.dataframe(svc.cardinalities().head(30), use_container_width=True)

    # ==================== 2) EXPLORATION ====================
    with tabs[1]:
        st.caption(f"{len(df):,} rows (before filters)")

        colA, colB = st.columns(2)
        h = svc.hist_rating()
        if not h.empty:
            with colA:
                st.plotly_chart(
                    px.bar(h, x="left", y="count", title="Ratings distribution"),
                    use_container_width=True,
                )
        h2 = svc.hist_review_len()
        if not h2.empty:
            with colB:
                st.plotly_chart(
                    px.bar(h2, x="left", y="count", title="Review length (chars)"),
                    use_container_width=True,
                )

        bm = svc.by_month()
        if not bm.empty:
            st.subheader("Trend over time")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.line(bm, x="month", y="n", title="Reviews per month"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    px.line(bm, x="month", y="mean_rating", title="Avg rating per month"),
                    use_container_width=True,
                )

            yr = svc.year_range()
            if yr:
                year = st.slider("Year (zoom)", yr[0], yr[1], value=int((yr[0] + yr[1]) // 2), key="zoom_year")
                oy = svc.one_year(year)
                c3, c4 = st.columns(2)
                with c3:
                    fig, ax = plt.subplots(figsize=(9, 3.8))
                    ax.plot(oy["month"], oy["n"], marker="o")
                    ax.set_title(f"Volume per month — {year}")
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                    st.pyplot(fig, clear_figure=True)
                with c4:
                    fig, ax = plt.subplots(figsize=(9, 3.8))
                    ax.plot(oy["month"], oy["mean_rating"], marker="o")
                    ax.set_title(f"Avg rating per month — {year}")
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                    st.pyplot(fig, clear_figure=True)

        st.subheader("Aggregations")
        au = svc.agg_by_user()
        if not au.empty:
            st.write("Top users (by #reviews):")
            st.dataframe(au.head(10), use_container_width=True)
        ar = svc.agg_by_recipe()
        if not ar.empty:
            st.write("Top recipes (by #reviews):")
            st.dataframe(ar.head(10), use_container_width=True)

    # ==================== 3) TABLE ====================
    with tabs[2]:
        year_opt = None
        yr = svc.year_range()
        if yr:
            year_opt = st.slider("Year filter", yr[0], yr[1], key="table_year")

        fdf = svc.apply_filters(
            rating_range=rating_range,
            min_len=min_len,
            year=year_opt,
        )
        st.caption(f"{len(fdf):,} rows after filters")
        cols = [c for c in ["user_id", "recipe_id", "date", "rating", "review"] if c in fdf.columns]
        st.dataframe(fdf.head(1000)[cols], hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Export CSV (filters)",
            fdf.to_csv(index=False).encode("utf-8"),
            "reviews_filtered.csv",
            "text/csv",
        )


if __name__ == "__main__":
    app()
