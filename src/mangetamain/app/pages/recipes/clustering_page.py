from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from mangetamain.app.app_utils.ui import use_global_ui
from mangetamain.core.dataset import RecipesDataset
from mangetamain.core.recipes_eda import RecipesEDAService

# our 4 helpers
from mangetamain.core.clustering import clustering_recipes as recipes_text
from mangetamain.core.clustering import clustering_ingredients as ingredients
from mangetamain.core.clustering import clustering_time_tags as time_tags
from mangetamain.core.clustering import clustering_nutrivalues as nutrivalues



@st.cache_resource(show_spinner=True)
def _load_recipes_df() -> pd.DataFrame:
    ds = RecipesDataset()
    return ds.load().copy()


def app() -> None:
    use_global_ui(
        page_title="Mangetamain — Clustering & Similarity",
        subtitle="Cluster recipes (TF-IDF + KMeans) and explore ingredient / time / nutrition similarity.",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
    )

    # ======== Data Loading =========
    # Recipes dataset already uploaded in app entrypoint (main.py)
    df = st.session_state["recipes"]
    recipes_eda_svc = RecipesEDAService()
    recipes_eda_svc.load(df, preprocess=False)

    # =========================================================
    # 1. TEXT CLUSTERING
    # =========================================================
    st.header("1. Text clustering (TF-IDF + KMeans)")

    df["text"] = (
        df.get("name", pd.Series([""] * len(df))).fillna("")
        + " "
        + df.get("ingredients", pd.Series([""] * len(df))).astype(str)
        + " "
        + df.get("description", pd.Series([""] * len(df))).fillna("")
    ).str.lower()

    col_k, col_mf = st.columns(2)
    with col_k:
        k = st.slider("Number of clusters (k)", 3, 12, 6)
    with col_mf:
        maxf = st.slider("TF-IDF max features", 2000, 20000, 8000, step=1000)

    @st.cache_resource(show_spinner=True)
    def _fit_text_model(corpus: pd.Series, k: int, maxf: int):
        return recipes_text.build_tfidf_kmeans(corpus, k=k, maxf=maxf)

    km, tfidf = _fit_text_model(df["text"], k, maxf)
    df_clust = recipes_text.assign_clusters(df, tfidf, km, text_col="text")

    XY = recipes_text.compute_2d(tfidf, df_clust["text"])
    df_clust["x"], df_clust["y"] = XY[:, 0], XY[:, 1]

    st.plotly_chart(
        px.scatter(
            df_clust.sample(min(4000, len(df_clust))),
            x="x",
            y="y",
            color="cluster",
            hover_name="name",
            opacity=0.7,
            title="2-D projection of recipe clusters",
        ),
        config={'width': 'stretch'},
    )

    st.subheader("Top terms per cluster")
    st.dataframe(recipes_text.top_terms_per_cluster(km, tfidf))

    st.markdown("---")

    # =========================================================
    # 2. INGREDIENT SIMILARITY
    # =========================================================
    st.header("2. Ingredient similarity")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ing_sample_n = st.number_input("Sample size", 50, 50000, 5000, step=50)
    with c2:
        ing_seed = st.number_input("Random seed", 0, 9999, 5, step=1)
    with c3:
        ing_top_n = st.slider("Top N similar", 1, 50, 10)

    @st.cache_data(show_spinner=True)
    def _sample_df(n: int, seed: int) -> pd.DataFrame:
        if n >= len(df):
            return df.reset_index(drop=True)
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    sampled_ing = _sample_df(int(ing_sample_n), int(ing_seed))

    @st.cache_data(show_spinner=True)
    def _build_ing_sim(sampled: pd.DataFrame) -> pd.DataFrame:
        return ingredients.build_ingredient_similarity(sampled)

    sim_ing = _build_ing_sim(sampled_ing)

    if sim_ing.empty:
        st.warning("No ingredients similarity could be built from the sample.")
    else:
        options = [
            f"{rid} — {str(name)[:80]}"
            for rid, name in zip(sampled_ing["id"], sampled_ing.get("name", pd.Series([None] * len(sampled_ing))))
        ]
        sel = st.selectbox("Choose a recipe from the sample", options, key="ing_sel")
        selected_id = int(str(sel).split(" — ")[0])

        st.subheader("Selected recipe")
        row = sampled_ing[sampled_ing["id"] == selected_id]
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"**{r.get('name', '—')}**  \nminutes: **{r.get('minutes', '—')}**, steps: **{r.get('n_steps', '—')}**")
            if "ingredients" in sampled_ing.columns:
                st.write(r.get("ingredients", "[]"))

        try:
            neigh = ingredients.find_similar_by_ingredients(sim_ing, selected_id, top_n=ing_top_n)
        except KeyError:
            neigh = pd.Series(dtype=float)

        if neigh.empty:
            st.info("No similar recipes found for this one.")
        else:
            neigh_df = neigh.reset_index().rename(columns={"index": "id", selected_id: "score"})
            neigh_df = neigh_df.merge(sampled_ing, on="id", how="left")
            show_cols = [c for c in ["id", "name", "minutes", "n_steps", "ingredients"] if c in neigh_df.columns]
            neigh_df = neigh_df[["id", "score"] + [c for c in show_cols if c != "id"]]
            st.dataframe(neigh_df)

    st.markdown("---")

    # =========================================================
    # 3. TIME-TAG SIMILARITY
    # =========================================================
    st.header("3. Time-tag similarity")

    tt1, tt2, tt3 = st.columns([2, 1, 1])
    with tt1:
        tt_sample_n = st.number_input("Sample size (time-tags)", 10, 50000, 15000, step=10)
    with tt2:
        tt_seed = st.number_input("Random seed (time-tags)", 0, 9999, 5, step=1)
    with tt3:
        tt_top_n = st.slider("Top N similar (time-tags)", 1, 20, 5)

    sampled_tt = _sample_df(int(tt_sample_n), int(tt_seed))

    @st.cache_data(show_spinner=True)
    def _build_time_sim(d: pd.DataFrame) -> pd.DataFrame:
        return time_tags.compute_time_tag_similarity(d)

    sim_time = _build_time_sim(sampled_tt)

    if sim_time.empty:
        st.warning("No time-tag tokens found in the sample.")
    else:
        opts_t = [
            f"{rid} — {str(name)[:80]}"
            for rid, name in zip(sampled_tt["id"], sampled_tt.get("name", pd.Series([None] * len(sampled_tt))))
        ]
        sel_t = st.selectbox("Choose a recipe (time-tags)", opts_t, key="tt_sel")
        selected_id_t = int(str(sel_t).split(" — ")[0])

        try:
            neigh_t = time_tags.get_similar_recipes(sim_time, selected_id_t, top_n=tt_top_n)
        except KeyError:
            neigh_t = pd.Series(dtype=float)

        if neigh_t.empty:
            st.info("No similar recipes found for this one (time-tags).")
        else:
            neigh_df_t = neigh_t.reset_index().rename(columns={"index": "id", selected_id_t: "score"})
            neigh_df_t = neigh_df_t.merge(sampled_tt, on="id", how="left")
            use_cols = [c for c in ["id", "name", "minutes", "n_steps", "tags"] if c in neigh_df_t.columns]
            neigh_df_t = neigh_df_t[["id", "score"] + [c for c in use_cols if c != "id"]]
            st.dataframe(neigh_df_t)

    st.markdown("---")

    # =========================================================
    # 4. NUTRITION SIMILARITY
    # =========================================================
    st.header("4. Nutrition similarity")

    nc1, nc2, nc3 = st.columns([2, 1, 1])
    with nc1:
        nut_sample_n = st.number_input("Sample size (nutrition)", 10, 50000, 10000, step=10)
    with nc2:
        nut_seed = st.number_input("Random seed (nutrition)", 0, 9999, 5, step=1)
    with nc3:
        nut_top_n = st.slider("Top N similar (nutrition)", 1, 50, 10)

    df_nut = _sample_df(int(nut_sample_n), int(nut_seed))
    df_nut_norm, nut_info = nutrivalues.normalize_nutrition_columns(df_nut, try_parse_nut_column=True)

    @st.cache_data(show_spinner=True)
    def _build_nut_sim(d: pd.DataFrame, seed: int) -> pd.DataFrame:
        return nutrivalues.build_nutri_similarity(d, random_state=seed)

    sim_nut = _build_nut_sim(df_nut_norm, int(nut_seed))

    if sim_nut.empty:
        st.warning("No usable nutrition columns in the sample.")
        st.write(nut_info)
    else:
        opts_n = [
            f"{rid} — {str(name)[:80]}"
            for rid, name in zip(df_nut_norm["id"], df_nut_norm.get("name", pd.Series([None] * len(df_nut_norm))))
        ]
        sel_n = st.selectbox("Choose a recipe (nutrition)", opts_n, key="nut_sel")
        selected_id_n = int(str(sel_n).split(" — ")[0])

        try:
            neigh_n = nutrivalues.find_similar_by_nutri(sim_nut, selected_id_n, top_n=nut_top_n)
        except KeyError:
            neigh_n = pd.Series(dtype=float)

        if neigh_n.empty:
            st.info("No similar recipes found for this one (nutrition).")
        else:
            neigh_df_n = neigh_n.reset_index().rename(columns={"index": "id", selected_id_n: "score"})
            neigh_df_n = neigh_df_n.merge(df_nut_norm, on="id", how="left")
            show_cols_n = [c for c in ["id", "name", "minutes", "n_steps"] + nutrivalues.NUTRI_COLS if c in neigh_df_n.columns]
            neigh_df_n = neigh_df_n[["id", "score"] + [c for c in show_cols_n if c != "id"]]
            st.dataframe(neigh_df_n.fillna("—"))


if __name__ == "__main__":
    app()
