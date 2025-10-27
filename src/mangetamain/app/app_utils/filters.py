from __future__ import annotations
import pandas as pd
import streamlit as st


def ensure_session_filters():
    st.session_state.setdefault(
        "filters",
        {
            "minutes": (0, 240),
            "steps": (0, 20),
            "include_tags": [],
            "exclude_tags": [],
            "include_ings": [],
        },
    )


def parse_tag_str(x: str | None) -> list[str]:
    if pd.isna(x):
        return []
    x = str(x)
    if x.startswith("["):  # format liste stringifiée
        try:
            import ast

            return [t.strip().strip("'\"") for t in ast.literal_eval(x)]
        except Exception:
            pass
    return [t.strip() for t in x.replace("|", ",").split(",") if t.strip()]


def apply_basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    f = st.session_state["filters"]
    m1 = df["minutes"].between(*f["minutes"], inclusive="both")
    m2 = df["n_steps"].between(*f["steps"], inclusive="both") | df["n_steps"].isna()
    res = df[m1 & m2].copy()

    # tags include/exclude (façon simple)
    if f["include_tags"] or f["exclude_tags"]:
        res["__tags_list"] = res["tags"].apply(parse_tag_str)
        if f["include_tags"]:
            res = res[
                res["__tags_list"].apply(
                    lambda L: any(t in L for t in f["include_tags"])
                )
            ]
        if f["exclude_tags"]:
            res = res[
                ~res["__tags_list"].apply(
                    lambda L: any(t in L for t in f["exclude_tags"])
                )
            ]
        res = res.drop(columns="__tags_list", errors="ignore")
    # ingrédients include (optionnel, matching naïf)
    if f["include_ings"]:
        res["__ings"] = res["ingredients"].astype(str).str.lower()
        for ing in f["include_ings"]:
            res = res[res["__ings"].str.contains(str(ing).lower(), na=False)]
        res = res.drop(columns="__ings", errors="ignore")
    return res
