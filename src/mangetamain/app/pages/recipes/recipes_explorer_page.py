# app/pages/recipes/recipes_explorer_page.py
from __future__ import annotations
import io
import plotly.express as px
import streamlit as st
import pandas as pd
from dateutil.relativedelta import relativedelta

from mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.app.app_utils.viz import kde_with_range, kde_plot
from mangetamain.core.recipes_eda import RecipesEDAService
from mangetamain.core.dataset import DatasetLoader


def format_period(date_min, date_max):
    """Formats the period between the minimum and maximum dates.

    Args:
        date_min (datetime.date): The minimum date.
        date_max (datetime.date): The maximum date.
    """
    if pd.notnull(date_min) and pd.notnull(date_max):
        diff_days = (date_max - date_min).days
        diff = relativedelta(date_max, date_min)

        # Dynamically format the period
        if diff_days < 60:
            diff_str = f"{diff_days} days"
        elif diff.years < 1:
            diff_str = f"{diff.months} months, {diff.days} days"
        else:
            diff_str = f"{diff.years} years, {diff.months} months"

        # Display cleanly in Streamlit
        return f"{date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')} ({diff_str})"
    else:
        return "N/A"
    
def minutes_to_period(minutes: int) -> str:
    MINUTES_IN_YEAR = 525600
    MINUTES_IN_MONTH = 43800
    MINUTES_IN_DAY = 1440
    MINUTES_IN_HOUR = 60

    years, rem = divmod(minutes, MINUTES_IN_YEAR)
    months, rem = divmod(rem, MINUTES_IN_MONTH)
    days, rem = divmod(rem, MINUTES_IN_DAY)
    hours, mins = divmod(rem, MINUTES_IN_HOUR)

    parts = []
    if years: parts.append(f"{years}y")
    if months: parts.append(f"{months}mo")
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if mins or not parts: parts.append(f"{mins}m")

    return " ".join(parts)

def add_filters(recipes_eda_svc: RecipesEDAService):
    m_lo, m_hi = recipes_eda_svc.minutes_range()
    s_lo, s_hi = recipes_eda_svc.nsteps_range()

    minutes_stats = recipes_eda_svc.minutes_stats()
    minutes_percentiles_25 = int(minutes_stats['25%'])
    minutes_percentiles_50 = int(minutes_stats['50%'])
    minutes_percentiles_75 = int(minutes_stats['75%'])
    minutes_percentiles_80 = int(minutes_stats['80%'])
    minutes_percentiles_90 = int(minutes_stats['90%'])
    minutes_percentiles_95 = int(minutes_stats['95%'])

    st.title(":gear: Filters")

    # Define your dataset statistics
    minutes_ranges = {
        f"All        ({minutes_to_period(m_lo)} - {minutes_to_period(m_hi)})": (m_lo, m_hi),
        f"Very short ({minutes_to_period(m_lo)} - {minutes_to_period(minutes_percentiles_25)})": (m_lo, minutes_percentiles_25),
        f"Short      ({minutes_to_period(minutes_percentiles_25)} - {minutes_to_period(minutes_percentiles_75)})": (minutes_percentiles_25, minutes_percentiles_75),
        f"Medium     ({minutes_to_period(minutes_percentiles_75)} - {minutes_to_period(minutes_percentiles_80)})": (minutes_percentiles_75, minutes_percentiles_80),
        f"Long       ({minutes_to_period(minutes_percentiles_80)} - {minutes_to_period(minutes_percentiles_90)})": (minutes_percentiles_80, minutes_percentiles_90),
        f"Very long  ({minutes_to_period(minutes_percentiles_90)} - {minutes_to_period(minutes_percentiles_95)})": (minutes_percentiles_90, minutes_percentiles_95),
        f"Extra long (> {minutes_to_period(minutes_percentiles_95)})": (minutes_percentiles_95, m_hi), 
        f"Custom range": None
    }

    # Show radio buttons
    selected_label = st.selectbox(
        "Select recipe duration range",
        options=list(minutes_ranges.keys()), 
        placeholder="Select recipe duration range"
    )
    # Get the corresponding numeric range
    # Determine numeric range
    if selected_label == "Custom range":
        st.info("Enter your custom range in minutes:")
        min_minutes = st.number_input("Min minutes", min_value=m_lo, max_value=m_hi, value=m_lo, step=1)
        max_minutes = st.number_input("Max minutes", min_value=min_minutes, max_value=m_hi, value=minutes_percentiles_90, step=1)
        if max_minutes < min_minutes:
            st.error("Max cannot be smaller than Min!")
            selected_range = None
        else:
            selected_range = (min_minutes, max_minutes)
    else:
        selected_range = minutes_ranges[selected_label]
    m_lo, m_hi = selected_range

    df_filtered = recipes_eda_svc.ds.df[recipes_eda_svc.ds.df["minutes"].between(*selected_range)]

    minutes_range = st.slider(
        "Select Minutes Range",
        min_value=m_lo, max_value=m_hi,
        value=(m_lo, m_hi),
    )
    df_filtered = df_filtered[df_filtered["minutes"].between(*minutes_range)]
    st.plotly_chart(
        kde_plot(df_filtered, "minutes"),
        config={'width': 'stretch'}
    )
    st.write(f"Selected range: {minutes_to_period(minutes_range[0])} → {minutes_to_period(minutes_range[1])}")

    # --- N_STEPS FILTER ---
    steps_range = st.slider(
        "Select n_steps Range",
        min_value=s_lo, max_value=s_hi,
        value=(s_lo, s_hi)
    )
    df_filtered = df_filtered[df_filtered["n_steps"].between(*steps_range)]
    st.plotly_chart(
        kde_plot(df_filtered, "n_steps"),
        config={'width': 'stretch'}
    )
    # inc_tags = [t.strip() for t in st.text_input("Include tags ( , )", "").split(",") if t.strip()]
    # inc_ings = [t.strip() for t in st.text_input("Contains ingredients ( , )", "").split(",") if t.strip()]

    df_filtered = recipes_eda_svc.apply_filters(
        minutes_range=minutes_range, steps_range=steps_range, 
        # include_tags=True, include_ings=True
    )

    return df_filtered

def app():
    use_global_ui(
        page_title="Mangetamain — Recipes Exploratory Data Analysis",
        subtitle="Explore and analyze the Food.com recipes dataset",
        logo="assets/mangetamain-logo.jpg", 
        logo_size_px=90, round_logo=True, wide=True,
    )

    # ======== Data Loading =========
    # Recipes dataset already uploaded in app entrypoint (main.py)
    recipes_df = st.session_state["recipes"]
    recipes_eda_svc = RecipesEDAService()
    recipes_eda_svc.load(recipes_df, preprocess=False)

    # Filters
    with st.sidebar:
        df_filtered = add_filters(recipes_eda_svc)

    df_columns = df_filtered.columns.tolist()

    # Reload the service with filtered data so downstream methods reflect filters
    recipes_eda_svc.load(df_filtered, preprocess=False)

    # ========= KPI Header =========
    unique_tags = recipes_eda_svc.get_unique("tags")
    unique_ings = recipes_eda_svc.get_unique("ingredients")
    cols= st.columns(
        spec=4, 
        gap="small", 
        border=True)
    cols[0].metric("Recipes", f"{len(df_filtered):,}")
    cols[1].metric("Columns", f"{len(df_columns)}")
    cols[2].metric("Unique Tags", f"{len(unique_tags):,}")
    cols[3].metric("Unique Ingredients", f"{len(unique_ings):,}")

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
    date_min = df_filtered["submitted"].dt.date.min()
    date_max = df_filtered["submitted"].dt.date.max()
    period = format_period(date_min, date_max)
    st.metric(
        "Period", 
        period, 
        border=True)
    
    with st.expander("👀 Preview"):
        st.dataframe(df_filtered.head(20))

    # --------- Tabs ----------
    tab1, tab2, tab3 = st.tabs([":broom: Quality", ":bar_chart: Exploration", ":page_facing_up: Table"])

    # ---- Quality ----
    with tab1:
        st.subheader("Schema & Completeness")
        # buf = io.StringIO(); df_filtered.info(buf=buf); st.text(buf.getvalue())
        st.dataframe(DatasetLoader.compute_schema(df_filtered))

        st.subheader("Missing Values")
        miss = recipes_eda_svc.na_counts()
        if miss.empty:
            st.write("No missing values detected in the dataset.")
        else:
            st.dataframe(miss)
            st.bar_chart(miss)

        st.subheader("Duplicates")
        dups = recipes_eda_svc.duplicates()
        if not dups:
            st.write("No duplicates found in the dataset.")
        else:
            for key, val in dups.items():
                if key != "full":
                    st.write(f"Duplicates on {key.split('_')} : **{val}**")
                else:
                    st.write(f"Duplicates (all columns) : **{val}**")
            
        st.subheader("Descriptive Statistics & Cardinalities")
        st.dataframe(recipes_eda_svc.numeric_desc())
        st.dataframe(recipes_eda_svc.cardinalities())

    # ---- Exploration ----
    with tab2:
        st.caption(f"{len(df_filtered):,} recipes after applying filters.")

        c1, c2 = st.columns(2)
        hmin = recipes_eda_svc.minutes_hist()
        if not hmin.empty:
            c1.plotly_chart(
                px.bar(hmin, x="left", y="count", title="Distribution of Minutes"), 
                config={"width": 'stretch'})
        hstp = recipes_eda_svc.steps_hist()
        if not hstp.empty:
            c2.plotly_chart(
                px.bar(hstp, x="left", y="count", title="Distribution of Steps"),
                config={"width": 'stretch'}
            )

        byy = recipes_eda_svc.by_year()
        if not byy.empty:
            st.plotly_chart(
                px.line(byy, x="year", y="n", title="Recipes by Year"), 
                config={"width": 'stretch'})

        top_ing = recipes_eda_svc.top_ingredients(30)
        if not top_ing.empty:
            st.subheader("Top Ingredients")
            st.plotly_chart(
                px.bar(top_ing.head(20), x="ingredient", y="count"), 
                config={"width": 'stretch'})

    # ---- Table ----
    with tab3:
        cols = [c for c in [
            "name", "minutes", "n_steps", "n_ingredients", "tags"
            ] if c in df_filtered.columns]
        st.dataframe(df_filtered[cols].head(1000), hide_index=True)
        st.download_button(":arrow_down: Export CSV (filtres)", df_filtered.to_csv(index=False).encode("utf-8"),
                        "recipes_filtered.csv", "text/csv")

if __name__ == "__main__":
    app()