# app/pages/recipes/recipes_explorer_page.py
from __future__ import annotations
import io
import plotly.express as px
import streamlit as st

from mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.core.recipes_eda import RecipesEDAService


def app():
    use_global_ui(
        page_title="Mangetamain — Recipes Exploratory Data Analysis",
        subtitle="Explore and analyze the Food.com recipes dataset",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
        wide=True,
    )

    # ======== Data Loading =========
    # Recipes dataset already uploaded in app entrypoint (main.py)
    recipes_df = st.session_state["recipes"]
    recipes_eda_svc = RecipesEDAService()
    recipes_eda_svc.load(recipes_df, preprocess=False)

    # # ========= Overview =========
    # st.subheader("Quick Overview")
    # st.plotly_chart(hist_minutes(recipes_df), width='stretch')

    # --------- Barre latérale : artefact + filtres ----------
    # with st.sidebar:
    #     st.header(":inbox_tray: Data Source")
    #     uploaded = st.file_uploader("Upload CSV/Parquet (optional)", type=["csv","parquet"])

    with st.sidebar:
        # st.header(":gear: Artefact")
        # if st.button(":arrows_counterclockwise: Generate Clean Recipes Artifact"):
        #     with st.spinner("Preprocessing..."):
        #         recipes_eda_svc.export_clean_min()
        #     st.success(f":microscope: Artifact regenerated successfully.")
        #     st.rerun()  # Refresh to load new artifact

        st.header(":gear: Filters")
        m_lo, m_hi = 0, int(recipes_df["minutes"].max(skipna=True) or 240)
        s_lo, s_hi = 0, int(recipes_df["n_steps"].max(skipna=True) or 20)
        minutes = st.slider("Minutes", m_lo, m_hi, (0, min(120, m_hi)))
        steps = st.slider("Steps", s_lo, s_hi, (0, min(12, s_hi)))
        inc_tags = [
            t.strip()
            for t in st.text_input("Include tags ( , )", "").split(",")
            if t.strip()
        ]
        inc_ings = [
            t.strip()
            for t in st.text_input("Contains ingredients ( , )", "").split(",")
            if t.strip()
        ]

    df_filtered = recipes_eda_svc.apply_filters(
        minutes=minutes, steps=steps, include_tags=inc_tags, include_ings=inc_ings
    )
    df_columns = df_filtered.columns.tolist()
    recipes_eda_svc.ds.df = df_filtered  # Update EDA service dataset

    # ========= KPI Header =========
    nb_cols = 2
    if inc_tags:
        nb_cols += 1
        unique_tags = recipes_eda_svc.get_unique("tags")
    if inc_ings:
        nb_cols += 1
        unique_ings = recipes_eda_svc.get_unique("ingredients")
    cols = st.columns(spec=nb_cols, gap="small", border=True)
    cols[0].metric("Recipes", f"{len(df_filtered):,}")
    cols[1].metric("Columns", f"{len(df_columns)}")
    for col in cols[2:]:
        if inc_tags:
            col.metric("Unique Tags", f"{len(unique_tags):,}")
        if inc_ings:
            col.metric("Unique Ingredients", f"{len(unique_ings):,}")

    # --------- KPIs ----------
    # Na per column
    if "issues" in st.session_state and "nan" in recipes_eda_svc.ds.issues:
        nan_dict = recipes_eda_svc.ds.issues["nan"]
    elif "nan" in recipes_eda_svc.ds.issues:
        nan_dict = recipes_eda_svc.ds.issues["nan"]
    else:
        nan_dict = {}

    if nan_dict:
        row = st.container(horizontal=True)
        with row:
            for col, na_val in nan_dict.items():
                if na_val > 0:
                    st.metric(f"{col} NA", na_val)

    # Period
    date_min = df_filtered["submitted"].min()
    date_max = df_filtered["submitted"].max()
    st.metric("Period", f"{date_min} → {date_max}", border=True)

    # --------- Tabs ----------
    tab1, tab2, tab3 = st.tabs(
        [":broom: Quality", ":bar_chart: Exploration", ":page_facing_up: Table"]
    )

    # ---- Quality ----
    with tab1:
        st.subheader("Schema & Completeness")
        with st.expander("👀 Preview / Schema"):
            st.dataframe(df_filtered.head(20), width="stretch")
            buf = io.StringIO()
            df_filtered.info(buf=buf)
            st.text(buf.getvalue())
            st.dataframe(recipes_eda_svc.schema(), width="stretch")

        st.subheader("Missing Values (Top 10)")
        miss = recipes_eda_svc.na_rate().head(10) * 100
        st.bar_chart(miss)

        st.subheader("Duplicates")
        dups = recipes_eda_svc.duplicates()
        for key, val in dups.items():
            if key != "full":
                st.write(f"Duplicates on {key.split('_')} : **{val}**")
            else:
                st.write(f"Duplicates (all columns) : **{val}**")

        st.subheader("Descriptive Statistics & Cardinalities")
        st.dataframe(recipes_eda_svc.numeric_desc().head(20), width="stretch")
        st.dataframe(recipes_eda_svc.cardinalities().head(30), width="stretch")

    # ---- Exploration ----
    with tab2:
        fdf = recipes_eda_svc.apply_filters(
            minutes=minutes, steps=steps, include_tags=inc_tags, include_ings=inc_ings
        )
        st.caption(f"{len(fdf):,} recipes after applying filters.")

        c1, c2 = st.columns(2)
        hmin = recipes_eda_svc.minutes_hist()
        if not hmin.empty:
            c1.plotly_chart(
                px.bar(hmin, x="left", y="count", title="Distribution of Minutes"),
                width="stretch",
            )
        hstp = recipes_eda_svc.steps_hist()
        if not hstp.empty:
            c2.plotly_chart(
                px.bar(hstp, x="left", y="count", title="Distribution of Steps"),
                width="stretch",
            )

        byy = recipes_eda_svc.by_year()
        if not byy.empty:
            st.plotly_chart(
                px.line(byy, x="year", y="n", title="Recipes by Year"), width="stretch"
            )

        top_ing = recipes_eda_svc.top_ingredients(30)
        if not top_ing.empty:
            st.subheader("Top Ingredients")
            st.plotly_chart(
                px.bar(top_ing.head(20), x="ingredient", y="count"), width="stretch"
            )

    # ---- Table ----
    with tab3:
        cols = [
            c
            for c in ["name", "minutes", "n_steps", "n_ingredients", "tags"]
            if c in df_filtered.columns
        ]
        view = recipes_eda_svc.apply_filters(
            minutes=minutes, steps=steps, include_tags=inc_tags, include_ings=inc_ings
        )
        st.dataframe(view[cols].head(1000), width="stretch", hide_index=True)
        st.download_button(
            ":arrow_down: Export CSV (filtres)",
            view.to_csv(index=False).encode("utf-8"),
            "recipes_filtered.csv",
            "text/csv",
        )


if __name__ == "__main__":
    app()
