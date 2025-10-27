import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app_utils.io import load_data
from app.app_utils.ui import use_global_ui

from core import clustering_recipes as cr


use_global_ui(
    page_title="Mangetamain — Similarité (time-tags)",
    logo="image/image.jpg",
    logo_size_px=90,
    round_logo=True,
    wide=True,
)


st.markdown(
    "This page computes recipe similarity using the time-related tags (e.g. '15-minutes').\nSelect a random sample, pick a recipe and see its nearest neighbours based on cosine similarity of time-tags."
)


@st.cache_data(show_spinner=True)
def load_and_sample(n: int, seed: int) -> pd.DataFrame:
    df = load_data()
    if n is None or n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


st.sidebar.header("Sampling & parameters")
sample_n = st.sidebar.number_input(
    "Sample size (random subset)", min_value=10, max_value=50000, value=15000, step=10
)
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=9999, value=42, step=1
)
top_n = st.sidebar.slider("Show top N similar recipes", 1, 20, 5)


with st.spinner("Loading and sampling data..."):
    sampled = load_and_sample(int(sample_n), int(seed))


@st.cache_data(show_spinner=True)
def build_similarity(df: pd.DataFrame) -> pd.DataFrame:
    # compute_time_tag_similarity expects the dataframe with 'id' and 'tags' (or time_tags_str)
    return cr.compute_time_tag_similarity(df, sample_n=None)


sim_df = build_similarity(sampled)

if sim_df.empty:
    st.warning(
        "No time-tags found in the sampled recipes. Try a larger sample or check that tags exist and contain time-related tokens."
    )
    st.stop()


# Selection UI: show friendly labels
options = [
    f"{rid} — {str(name)[:80]}"
    for rid, name in zip(
        sampled["id"], sampled.get("name", pd.Series([None] * len(sampled)))
    )
]
sel = st.selectbox("Choose a recipe (from the sampled set)", options)
selected_id = int(str(sel).split(" — ")[0])

st.subheader("Selected recipe")
sel_row = sampled[sampled["id"] == selected_id]
if not sel_row.empty:
    r = sel_row.iloc[0]
    st.markdown(
        f"**{r.get('name','—')}**  \n                minutes: **{r.get('minutes','—')}**, steps: **{r.get('n_steps','—')}**"
    )
    if "tags" in sampled.columns:
        st.markdown(f"tags: {r.get('tags','[]')}")


with st.spinner("Computing similar recipes..."):
    try:
        neigh = cr.get_similar_recipes(sim_df, selected_id, top_n=top_n)
    except KeyError:
        st.error(
            "Selected recipe is not present in the similarity matrix (maybe it has no time-tags)."
        )
        st.stop()

if neigh.empty:
    st.info("No similar recipes found for this selection.")
else:
    # Join with sampled metadata
    neigh_df = neigh.reset_index().rename(columns={"index": "id", selected_id: "score"})
    neigh_df = neigh_df.merge(sampled, on="id", how="left")
    display_cols = [
        c for c in ["id", "name", "minutes", "n_steps", "tags"] if c in neigh_df.columns
    ]
    neigh_df = neigh_df[["id", "score"] + [c for c in display_cols if c not in ("id")]]
    st.subheader(f"Top {len(neigh_df)} similar recipes")
    st.dataframe(neigh_df.fillna("—"), use_container_width=True)

    # Optional: allow to inspect one of the neighbours
    if not neigh_df.empty:
        inspect_id = st.selectbox("Inspect neighbour", neigh_df["id"].astype(str))
        insp_row = neigh_df[neigh_df["id"].astype(str) == str(inspect_id)].iloc[0]
        st.markdown(f"### {insp_row.get('name','—')}")
        st.write(
            {
                k: insp_row.get(k)
                for k in ["minutes", "n_steps", "tags"]
                if k in insp_row.index
            }
        )
