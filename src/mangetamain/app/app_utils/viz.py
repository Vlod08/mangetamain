# app/app_utils/viz.py
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def scatter_time_steps(df: pd.DataFrame):
    return px.scatter(
        df,
        x="minutes",
        y="n_steps",
        hover_name="name",
        size="n_ingredients",
        opacity=0.6,
    )


def hist_minutes(df: pd.DataFrame):
    return px.histogram(df, x="minutes", nbins=40)


def hist_steps(df: pd.DataFrame):
    return px.histogram(df, x="n_steps", nbins=20)


def bar_top_counts(series: pd.Series, top=15, title=""):
    vc = series.value_counts().head(top).reset_index()
    vc.columns = ["value", "count"]
    fig = px.bar(vc, x="value", y="count", title=title)
    fig.update_layout(xaxis_tickangle=-30)
    return fig


# --- Cache the KDE computation for performance ---
@st.cache_data
def compute_kde(df, column, num_points=200):
    data = df[column].dropna().values
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), num_points)
    y = kde(x)
    return x, y


def kde_plot(df, column, color="#1f77b4"):
    x, y = compute_kde(df, column)

    fig = go.Figure()

    # Full KDE curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="tozeroy",
            mode="lines",
            line=dict(color=color),
            fillcolor=color.replace("1.0", "0.4"),  # slightly transparent
            name=column,
        )
    )

    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title=column,
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
    )

    return fig


# --- Function to create shaded KDE plot quickly ---
def kde_with_range(df, column, value_range, color="#1f77b4"):
    x, y = compute_kde(df, column)

    fig = go.Figure()

    # Full KDE curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="tozeroy",
            mode="lines",
            line=dict(color=color),
            fillcolor=color.replace("1.0", "0.4"),  # slightly transparent
            name=column,
        )
    )

    # Shade outside range (gray)
    fig.add_vrect(
        x0=x.min(), x1=value_range[0], fillcolor="gray", opacity=0.25, line_width=0
    )
    fig.add_vrect(
        x0=value_range[1], x1=x.max(), fillcolor="gray", opacity=0.25, line_width=0
    )

    # Range lines
    fig.add_vline(x=value_range[0], line_dash="dash", line_color="red")
    fig.add_vline(x=value_range[1], line_dash="dash", line_color="red")

    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title=column,
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
    )

    return fig
