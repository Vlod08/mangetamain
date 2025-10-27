import plotly.express as px
import pandas as pd


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
