from pathlib import Path
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app_utils.ui import use_global_ui
from app.app_utils.io import load_data
from core import clustering_ingredients as ci


use_global_ui(
    page_title="Mangetamain — Similarité ingrédients",
    logo="image/image.jpg",
    logo_size_px=90,
    round_logo=True,
    wide=True,
)

st.markdown(
    "Compute recipe similarity using ingredients only (Count vectorization + cosine similarity). Pick a random sample, select a recipe and see the most similar recipes by ingredients."
)


@st.cache_data(show_spinner=True)
def sample_recipes(n: int, seed: int) -> pd.DataFrame:
    df = load_data()
    if n is None or n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


st.sidebar.header("Sampling & options")
sample_n = st.sidebar.number_input(
    "Sample size", min_value=50, max_value=50000, value=5000, step=50
)
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=9999, value=42, step=1
)
top_n = st.sidebar.slider("Show top N similar recipes", 1, 50, 10)

with st.spinner("Loading and sampling recipes..."):
    df = sample_recipes(int(sample_n), int(seed))


@st.cache_data(show_spinner=True)
def build_sim(df: pd.DataFrame) -> pd.DataFrame:
    return ci.build_ingredient_similarity(df, sample_n=None)


sim = build_sim(df)

if sim.empty:
    st.warning(
        "No ingredient tokens found in the sampled data. Try increasing the sample size or check the 'ingredients' column."
    )
    st.stop()

# Selection UI
options = [
    f"{rid} — {str(name)[:80]}"
    for rid, name in zip(df["id"], df.get("name", pd.Series([None] * len(df))))
]
sel = st.selectbox("Choose a recipe (from the sampled set)", options)
selected_id = int(str(sel).split(" — ")[0])

st.subheader("Selected recipe")
sel_row = df[df["id"] == selected_id]
if not sel_row.empty:
    r = sel_row.iloc[0]
    st.markdown(
        f"**{r.get('name','—')}**  \nminutes: **{r.get('minutes','—')}**, steps: **{r.get('n_steps','—')}**"
    )
    if "ingredients" in df.columns:
        st.markdown(f"ingredients: {r.get('ingredients','[]')}")

with st.spinner("Computing nearest neighbours (ingredients)..."):
    try:
        neighbours = ci.find_similar_by_ingredients(sim, selected_id, top_n=top_n)
    except KeyError:
        st.error(
            "Selected recipe not found in similarity matrix; it may have empty ingredients."
        )
        st.stop()

if neighbours.empty:
    st.info("No similar recipes found for this selection.")
else:
    neigh_df = neighbours.reset_index().rename(
        columns={"index": "id", selected_id: "score"}
    )
    neigh_df = neigh_df.merge(df, on="id", how="left")
    display_cols = [
        c
        for c in ["id", "name", "minutes", "n_steps", "ingredients"]
        if c in neigh_df.columns
    ]
    neigh_df = neigh_df[["id", "score"] + [c for c in display_cols if c not in ("id")]]
    st.subheader(f"Top {len(neigh_df)} similar recipes (ingredients)")
    st.dataframe(neigh_df.fillna("—"), use_container_width=True)

    if not neigh_df.empty:
        inspect_id = st.selectbox("Inspect neighbour", neigh_df["id"].astype(str))
        insp_row = neigh_df[neigh_df["id"].astype(str) == str(inspect_id)].iloc[0]
        st.markdown(f"### {insp_row.get('name','—')}")
        st.write(
            {
                k: insp_row.get(k)
                for k in ["minutes", "n_steps", "ingredients"]
                if k in insp_row.index
            }
        )
