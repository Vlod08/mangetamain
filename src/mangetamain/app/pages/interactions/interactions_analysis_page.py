# app/pages/interactions/interactions_analysis_page.py
from __future__ import annotations
import streamlit as st
import seaborn as sns
import plotly.express as px

from app.app_utils.ui import use_global_ui
from core.interactions_eda import InteractionsEDAService

# Optional fallback loader if the DF isn't in session_state
try:
    from core.dataset import get_interactions_loader
except Exception:
    get_interactions_loader = None

sns.set_theme()


INTRO_MD = """
## Advanced Interactions — What this page does

This page analyzes review **interactions** to understand:
- **Temporal dynamics**: monthly evolution, rolling smoothing, YoY growth, seasonality, anomalies, weekly/hourly patterns.
- **User bias**: how strict or generous users are, and whether heavy reviewers behave differently.
- **Text ↔ rating**: which tokens are most associated with each rating bucket.

**How to read the charts**
- *Time Series*: start with **Reviews per month**, then look at **Rolling 3** for smoothed trends.  
  **YoY** compares each month vs the same month last year (growth rate).  
  **Seasonal decomposition** splits the series into **trend / seasonal / residual** components.  
  **Anomalies** flag outliers via a z-score threshold.
- *User Bias*: the **histogram of user means** reveals strict vs generous raters;  
  **volume vs mean** shows whether heavy contributors lean higher/lower than average.
- *Text ↔ Rating*: “Top tokens by rounded rating” surfaces vocabulary linked to perceived quality.

> Tip: Install `statsmodels` to enable the OLS trendline and seasonal decomposition.
"""


def _load_interactions_df():
    if "interactions" in st.session_state:
        return st.session_state["interactions"]
    if get_interactions_loader is not None:
        try:
            return get_interactions_loader().df
        except Exception:
            pass
    return None


def _has_method(obj, name: str) -> bool:
    return hasattr(obj, name) and callable(getattr(obj, name))


def app():
    # ---------- Header ----------
    use_global_ui(
        page_title="Mangetamain — Interactions: Advanced Analysis",
        subtitle="Time series (monthly/seasonality/YoY), user bias, and text ↔ rating.",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
    )

    # ---------- Intro / Read me ----------
    with st.expander("🧭 What’s in here? (read me)", expanded=True):
        st.markdown(INTRO_MD)

    # ---------- Data ----------
    df = _load_interactions_df()
    if df is None or df.empty:
        st.warning(
            "The `interactions` dataset is not available. "
            "Load it in the entrypoint (`main.py`) or enable `get_interactions_loader()`."
        )
        return

    svc = InteractionsEDAService()
    svc.load(df, preprocess=False)

    # ---------- KPIs ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{len(df):,}".replace(",", " "))
    if "date" in df.columns and df["date"].notna().any():
        dmin, dmax = df["date"].min(), df["date"].max()
        c2.metric(
            "Period",
            f"{getattr(dmin, 'date', lambda: dmin)()} → {getattr(dmax, 'date', lambda: dmax)()}",
        )
    else:
        c2.metric("Period", "—")
    c3.metric(
        "Unique users",
        f"{df['user_id'].nunique():,}".replace(",", " ") if "user_id" in df else "—",
    )
    c4.metric(
        "Unique recipes",
        (
            f"{df['recipe_id'].nunique():,}".replace(",", " ")
            if "recipe_id" in df
            else "—"
        ),
    )

    tabs = st.tabs(["🗓️ Time Series", "👥 User Bias", "📝 Text ↔ Rating"])

    # ─────────────────────────────
    # 1) TIME SERIES
    # ─────────────────────────────
    with tabs[0]:
        st.subheader("Monthly evolution (raw)")
        st.markdown(
            "_Baseline monthly volume and mean rating. Use this to spot growth and broad seasonality._"
        )
        if _has_method(svc, "monthly_series"):
            bm = svc.monthly_series()
            if bm.empty:
                st.info("No usable `date` column.")
            else:
                a, b = st.columns(2)
                with a:
                    st.plotly_chart(
                        px.line(bm, x="month", y="n", title="Reviews per month"),
                        config={"width": "stretch"},
                    )
                    st.caption(
                        "Monthly count of reviews. Look for trend breaks and seasonal peaks."
                    )
                with b:
                    st.plotly_chart(
                        px.line(
                            bm,
                            x="month",
                            y="mean_rating",
                            title="Average rating per month",
                        ),
                        config={"width": "stretch"},
                    )
                    st.caption(
                        "Monthly average rating. Watch for drifts in perceived quality."
                    )

        st.subheader("Smoothing (3-month rolling)")
        st.markdown("_Reduces noise; highlights medium-term trend._")
        if _has_method(svc, "monthly_rolling"):
            roll = svc.monthly_rolling(window=3)
            if not roll.empty:
                c, d = st.columns(2)
                with c:
                    st.plotly_chart(
                        px.line(roll, x="month", y="n_roll3", title="N (rolling 3)"),
                        config={"width": "stretch"},
                    )
                    st.caption(
                        "3-month rolling mean of monthly counts — smoother trend."
                    )
                with d:
                    st.plotly_chart(
                        px.line(
                            roll,
                            x="month",
                            y="mean_rating_roll3",
                            title="Rating (rolling 3)",
                        ),
                        config={"width": "stretch"},
                    )
                    st.caption(
                        "3-month rolling average of ratings — removes short-term fluctuations."
                    )

        st.subheader("Year-over-Year growth (YoY)")
        st.markdown("_Month vs same month last year. Positive means growth._")
        if _has_method(svc, "monthly_yoy"):
            yoy = svc.monthly_yoy()
            if not yoy.empty and "n_yoy" in yoy:
                fig = px.line(yoy, x="month", y="n_yoy", title="YoY N (t vs t-12)")
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, config={"width": "stretch"})
                st.caption(
                    "YoY growth in monthly counts. Values > 0 indicate expansion."
                )

        st.subheader("Seasonal decomposition")
        st.markdown(
            "_Decomposes the series into trend, seasonality, residual. Requires `statsmodels`._"
        )
        if _has_method(svc, "seasonal_decompose_monthly"):
            dec = svc.seasonal_decompose_monthly()
            if dec.empty:
                st.info("Install `statsmodels` or provide more data for decomposition.")
            else:
                st.plotly_chart(
                    px.line(
                        dec,
                        x="month",
                        y="value",
                        color="part",
                        title="Decomposition (trend / seasonal / resid)",
                    ),
                    config={"width": "stretch"},
                )
                st.caption(
                    "Helps separate long-term trend from recurring seasonal patterns and noise."
                )

        st.subheader("Anomalies (z-score)")
        st.markdown("_Flags outlier months with unusually high/low counts._")
        if _has_method(svc, "monthly_anomalies"):
            an = svc.monthly_anomalies(z_thresh=2.5)
            if not an.empty:
                st.plotly_chart(
                    px.scatter(
                        an,
                        x="month",
                        y="n",
                        color="is_anomaly",
                        title="Anomalies on monthly N (|z| ≥ 2.5)",
                        hover_data=["z_n"],
                    ),
                    config={"width": "stretch"},
                )
                st.caption(
                    "Points tagged `True` are statistical outliers (absolute z-score ≥ 2.5)."
                )

        st.subheader("Weekly & hourly profile")
        st.markdown(
            "_Activity by weekday and hour — useful for scheduling and UX assumptions._"
        )
        if _has_method(svc, "weekday_profile"):
            wk = svc.weekday_profile()
            if not wk.empty:
                wk = wk.copy()
                wk["weekday"] = wk["wk"].map(
                    {
                        0: "Mon",
                        1: "Tue",
                        2: "Wed",
                        3: "Thu",
                        4: "Fri",
                        5: "Sat",
                        6: "Sun",
                    }
                )
                st.plotly_chart(
                    px.bar(wk, x="weekday", y="n", title="Volume by weekday"),
                    config={"width": "stretch"},
                )
                st.caption("Which weekdays are busiest. Helps spot weekly cyclicality.")
        if _has_method(svc, "weekday_hour_heat"):
            mat = svc.weekday_hour_heat()
            if not mat.empty:
                st.plotly_chart(
                    px.density_heatmap(
                        mat,
                        x="h",
                        y="wk",
                        z="n",
                        nbinsx=24,
                        nbinsy=7,
                        title="Heatmap (hour × weekday)",
                    ),
                    config={"width": "stretch"},
                )
                st.caption(
                    "Heatmap of activity by hour (x) and weekday (y). Hot spots indicate high engagement windows."
                )

        st.subheader("User cohorts (overview)")
        st.markdown("_Cohorts by first interaction month; age is months since cohort._")
        if _has_method(svc, "cohorts_users"):
            coh = svc.cohorts_users()
            if not coh.empty:
                coh_small = coh[(coh["age"] >= 0) & (coh["age"] <= 12)].copy()
                st.plotly_chart(
                    px.imshow(
                        coh_small.pivot(
                            index="cohort", columns="age", values="n"
                        ).fillna(0),
                        aspect="auto",
                        title="Cohorts (0→12 months) — #reviews",
                    ),
                    config={"width": "stretch"},
                )
                st.caption(
                    "Simple cohort grid (first-interaction month × months since). Useful for retention-like patterns."
                )

    # ─────────────────────────────
    # 2) USER BIAS
    # ─────────────────────────────
    with tabs[1]:
        st.subheader("User bias")
        st.markdown(
            "_How strict/generous users are, and whether heavy reviewers behave differently._"
        )
        if _has_method(svc, "user_bias"):
            ub = svc.user_bias()
            if ub.empty:
                st.info("`user_id`/`rating` columns missing.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(ub.head(20))
                    st.caption(
                        "Sample of per-user stats: review count, mean, and median rating."
                    )
                with col2:
                    st.plotly_chart(
                        px.histogram(
                            ub,
                            x="mean",
                            nbins=25,
                            title="Distribution of per-user means",
                        ),
                        config={"width": "stretch"},
                    )
                    st.caption(
                        "Spread of user averages — wide spread suggests strong rater heterogeneity."
                    )
                st.plotly_chart(
                    px.scatter(
                        ub,
                        x="n",
                        y="mean",
                        title="Review volume vs mean",
                        hover_data=["median"],
                    ),
                    config={"width": "stretch"},
                )
                st.caption(
                    "Do heavy contributors rate higher/lower than average? Check the right side of the chart."
                )

        if _has_method(svc, "rating_vs_length"):
            rvl = svc.rating_vs_length()
            if not rvl.empty:
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
                st.plotly_chart(fig, config={"width": "stretch"})
                st.caption(
                    "Each dot is a review. OLS line (if enabled) shows the average relationship."
                )

    # ─────────────────────────────
    # 3) TEXT ↔ RATING
    # ─────────────────────────────
    with tabs[2]:
        st.subheader("Text ↔ rating (top tokens by rounded rating)")
        st.markdown("_Vocabulary that co-occurs with each rounded rating bucket._")
        if _has_method(svc, "tokens_by_rating"):
            tbr = svc.tokens_by_rating(k=20)
            if tbr.empty:
                st.info("`review`/`rating` columns missing.")
            else:
                st.dataframe(tbr.head(100))
                st.caption(
                    "Top 100 rows for quick inspection; use the bar chart to compare buckets."
                )
                st.plotly_chart(
                    px.bar(
                        tbr,
                        x="token",
                        y="count",
                        color="rating",
                        barmode="group",
                        title="Top tokens by rounded rating",
                    ),
                    config={"width": "stretch"},
                )
                st.caption(
                    "Compare tokens across rating buckets to spot words tied to high/low perceived quality."
                )


if __name__ == "__main__":
    app()
