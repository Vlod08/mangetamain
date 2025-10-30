# app/pages/recipes/clustering_page.py
import streamlit as st
import pandas as pd
import plotly.express as px

from app.app_utils.ui import use_global_ui
from core.clustering import clustering_recipes as cr
from core.clustering import clustering_ingredients as ci
from core.clustering import clustering_time_tags as ctt
from core.clustering import clustering_nutrivalues as cn
from core.recipes_eda import RecipesEDAService


def app():
    use_global_ui(
        "Mangetamain —  Clustering (TF-IDF + KMeans)",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True, subtitle=None, wide=True
        )

    # --- Clustering section (TF-IDF + KMeans)
    st.header("Clustering (TF-IDF + KMeans)")

    recipes_eda_svc = RecipesEDAService()

    # prepare data and text corpus for clustering
    df_clust = recipes_eda_svc.ds.df.copy()
    df_clust["text"] = (df_clust.get("name", pd.Series([""] * len(df_clust))).fillna("")
                + " " + df_clust.get("ingredients", pd.Series([""] * len(df_clust))).astype(str)
                + " " + df_clust.get("description", pd.Series([""] * len(df_clust))).fillna("")).str.lower()

    k = st.slider("Number of clusters (k)", 3, 12, 6)
    maxf = st.slider("Max features TF-IDF", 2000, 20000, 8000, step=1000)

    @st.cache_resource(show_spinner=True)
    def build_model(corpus: pd.Series, k: int, maxf: int):
        return cr.build_tfidf_kmeans(corpus, k=k, maxf=maxf)

    km, tfidf = build_model(df_clust["text"], k, maxf)
    df_clust = cr.assign_clusters(df_clust, tfidf, km, text_col="text")

    # 2D via TruncatedSVD
    XY = cr.compute_2d(tfidf, df_clust["text"], n_components=2)
    df_clust["x"], df_clust["y"] = XY[:, 0], XY[:, 1]

    fig = px.scatter(df_clust.sample(min(4000, len(df_clust))), x="x", y="y", color="cluster", hover_name="name", opacity=0.7)
    st.plotly_chart(fig, config={"width": 'stretch'})

    st.subheader("Top mots par cluster")
    st.dataframe(cr.top_terms_per_cluster(km, tfidf, topn=8), hide_index=True)


    # visual separator
    st.markdown("---")

    # --- Ingredients-only similarity section (previously page 09)
    st.header("Ingredients Similarity")
    st.markdown("Calculate recipe similarity using only ingredients (Count vectorization + cosine similarity). Choose a random sample, select a recipe, and display the most similar recipes based on ingredients.")

    @st.cache_data(show_spinner=True)
    def sample_recipes(n: int, seed: int) -> pd.DataFrame:
        df = recipes_eda_svc.ds.df.copy()
        if n is None or n <= 0 or n >= len(df):
            return df.reset_index(drop=True)
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    # Sampling & options placed inside the ingredients section (previously in sidebar)
    st.subheader("Sampling & options")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sample_n = st.number_input("Sample size", min_value=50, max_value=50000, value=5000, step=50, key="ing_sample_n")
    with c2:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=5, step=1, key="ing_seed")
    with c3:
        top_n = st.slider("Show top N similar recipes", 1, 50, 10, key="ing_top_n")

    with st.spinner("Loading and sampling recipes..."):
        df_sim = sample_recipes(int(sample_n), int(seed))

    @st.cache_data(show_spinner=True)
    def build_sim(df: pd.DataFrame) -> pd.DataFrame:
        return ci.build_ingredient_similarity(df, sample_n=None)

    sim = build_sim(df_sim)

    if sim.empty:
        st.warning("No ingredient tokens found in the sample. Try increasing the sample size or check the 'ingredients' column.")
    else:
        # Selection UI
        options = [f"{rid} — {str(name)[:80]}" for rid, name in zip(df_sim["id"], df_sim.get("name", pd.Series([None]*len(df_sim))))]
        sel = st.selectbox("Choose a recipe (from the sample)", options, key="ing_select")
        selected_id = int(str(sel).split(" — ")[0])

        st.subheader("Selected recipe")
        sel_row = df_sim[df_sim["id"] == selected_id]
        if not sel_row.empty:
            r = sel_row.iloc[0]
            st.markdown(f"**{r.get('name','—')}**  \nminutes : **{r.get('minutes','—')}**, steps : **{r.get('n_steps','—')}**")
            if "ingredients" in df_sim.columns:
                st.markdown(f"Ingredients : {r.get('ingredients','[]')}")

        with st.spinner("Calculating nearest neighbours (ingredients)..."):
            try:
                neighbours = ci.find_similar_by_ingredients(sim, selected_id, top_n=top_n)
            except KeyError:
                st.error("Selected recipe not found in similarity matrix; it may have empty ingredients.")
                neighbours = pd.Series(dtype=float)

        if neighbours.empty:
            st.info("No similar recipes found for this selection.")
        else:
            neigh_df = neighbours.reset_index().rename(columns={"index": "id", selected_id: "score"})
            neigh_df = neigh_df.merge(df_sim, on="id", how="left")
            display_cols = [c for c in ["id", "name", "minutes", "n_steps", "ingredients"] if c in neigh_df.columns]
            neigh_df = neigh_df[["id", "score"] + [c for c in display_cols if c not in ("id")]]
            st.subheader(f"Top {len(neigh_df)} similar recipes (ingredients)")
            st.dataframe(neigh_df.fillna("—"))

            if not neigh_df.empty:
                inspect_id = st.selectbox("Inspect neighbour", neigh_df["id"].astype(str), key="ing_inspect")
                insp_row = neigh_df[neigh_df["id"].astype(str) == str(inspect_id)].iloc[0]
                st.markdown(f"### {insp_row.get('name','—')}")
                st.write({k: insp_row.get(k) for k in ["minutes", "n_steps", "ingredients"] if k in insp_row.index})


    # visual separator for time-tags section
    st.markdown("---")

    # --- Time-tags similarity section (from previous page 08)
    st.header("Similarity (time tags)")
    st.markdown("This section calculates the similarity of recipes using time-related tags (e.g. '15-minutes'). Select a random sample, choose a recipe, and display its nearest neighbours based on the cosine similarity of the time tags.")


    @st.cache_data(show_spinner=True)
    def load_and_sample_tags(n: int, seed: int) -> pd.DataFrame:
        df = recipes_eda_svc.ds.df.copy()
        if n is None or n <= 0 or n >= len(df):
            return df.reset_index(drop=True)
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


    # Sampling controls for time tags (inline)
    st.subheader("Sampling and parameters (time tags)")
    tc1, tc2, tc3 = st.columns([2, 1, 1])
    with tc1:
        sample_n_tags = st.number_input("Sample size (random subset)", min_value=10, max_value=50000, value=15000, step=10, key="tags_sample_n")
    with tc2:
        seed_tags = st.number_input("Random seed", min_value=0, max_value=9999, value=5, step=1, key="tags_seed")
    with tc3:
        top_n_tags = st.slider("Show top N similar recipes", 1, 20, 5, key="tags_top_n")

    with st.spinner("Loading and sampling data (time tags)..."):
        sampled_tags = load_and_sample_tags(int(sample_n_tags), int(seed_tags))


    @st.cache_data(show_spinner=True)
    def build_similarity_time_tags(df: pd.DataFrame) -> pd.DataFrame:
        # compute_time_tag_similarity expects the dataframe with 'id' and 'tags' (or time_tags_str)
        return ctt.compute_time_tag_similarity(df, sample_n=None)


    sim_time = build_similarity_time_tags(sampled_tags)

    if sim_time.empty:
        st.warning("No time tag found in the sampled recipes. Try a larger sample or check that the 'tags' column contains time tokens.")
    else:
        # Selection UI: show friendly labels
        options_t = [f"{rid} — {str(name)[:80]}" for rid, name in zip(sampled_tags["id"], sampled_tags.get("name", pd.Series([None]*len(sampled_tags))))]
        sel_t = st.selectbox("Choose a recipe (from the sample)", options_t, key="tags_select")
        selected_id_t = int(str(sel_t).split(" — ")[0])
        st.subheader("Selected recipe (time tags)")
        sel_row_t = sampled_tags[sampled_tags["id"] == selected_id_t]
        if not sel_row_t.empty:
            r = sel_row_t.iloc[0]
            st.markdown(f"**{r.get('name','—')}**  \nminutes : **{r.get('minutes','—')}**, steps : **{r.get('n_steps','—')}**")
            if "tags" in sampled_tags.columns:
                st.markdown(f"Tags : {r.get('tags','[]')}")

        with st.spinner("Loading similar recipes (time tags)..."):
            try:
                neigh_t = ctt.get_similar_recipes(sim_time, selected_id_t, top_n=top_n_tags)
            except KeyError:
                st.error("The selected recipe is not present in the similarity matrix (it may not have time tags).")
                st.stop()

        if neigh_t.empty:
            st.info("No similar recipe found for this selection.")
        else:
            # Join with sampled metadata
            neigh_df_t = neigh_t.reset_index().rename(columns={"index": "id", selected_id_t: "score"})
            neigh_df_t = neigh_df_t.merge(sampled_tags, on="id", how="left")
            display_cols_t = [c for c in ["id", "name", "minutes", "n_steps", "tags"] if c in neigh_df_t.columns]
            neigh_df_t = neigh_df_t[["id", "score"] + [c for c in display_cols_t if c not in ("id")]]
            st.subheader(f"Top {len(neigh_df_t)} similar recipes (time tags)")
            st.dataframe(neigh_df_t.fillna("—"))

            # Optional: allow to inspect one of the neighbours
            if not neigh_df_t.empty:
                inspect_id_t = st.selectbox("Inspect neighbour (time tags)", neigh_df_t["id"].astype(str), key="tags_inspect")
                insp_row_t = neigh_df_t[neigh_df_t["id"].astype(str) == str(inspect_id_t)].iloc[0]
                st.markdown(f"### {insp_row_t.get('name','—')}")
                st.write({k: insp_row_t.get(k) for k in ["minutes", "n_steps", "tags"] if k in insp_row_t.index})


    # visual separator for nutrivalues section
    st.markdown("---")

    # --- Nutritional similarity section
    st.header("Nutritional Similarity")
    st.markdown("Calculate the similarity of recipes using nutritional values. The module attempts to normalize nutritional columns (mapping alternative names or parsing the 'nutrition' column) before computing similarity.")

    # Reuse the sample_recipes function defined above
    st.subheader("Sampling & Parameters (Nutrients)")
    nc1, nc2, nc3 = st.columns([2, 1, 1])
    with nc1:
        nutri_sample_n = st.number_input("Sample Size", min_value=10, max_value=50000, value=10000, step=10, key="nutri_sample_n")
    with nc2:
        nutri_seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=5, step=1, key="nutri_seed")
    with nc3:
        nutri_top_n = st.slider("Show Top N Similar Recipes", 1, 50, 10, key="nutri_top_n")


    with st.spinner("Loading and sampling recipes (nutrients)..."):
        df_nut = sample_recipes(int(nutri_sample_n), int(nutri_seed))

        # normalize the sampled dataframe so we can both compute similarity and
        # display the nutrition columns alongside the metadata
        df_nut_norm, nut_info = cn.normalize_nutrition_columns(df_nut, try_parse_nut_column=True)

        # build similarity on the normalized dataframe
        sim_nut = cn.build_nutri_similarity(df_nut_norm, sample_n=None, random_state=int(nutri_seed))

    if sim_nut.empty:
        st.warning("No usable nutritional data found in the sample.")
        st.markdown("**Diagnostics (nutrition normalization):**")
        st.write(nut_info if isinstance(nut_info, dict) else {})
        # show some nutritional columns if present
        present = nut_info.get("present", []) if isinstance(nut_info, dict) else [c for c in cn.NUTRI_COLS if c in df_nut_norm.columns]
        alt_present = nut_info.get("alt_present", []) if isinstance(nut_info, dict) else []
        cols_to_check = present + alt_present
        if cols_to_check:
            counts = df_nut_norm[cols_to_check].notna().sum().to_dict()
            st.write("Non-null counts per column (in the sample):", counts)
            st.dataframe(df_nut_norm[cols_to_check].head(10).fillna("—"))
    else:
        options_n = [f"{rid} — {str(name)[:80]}" for rid, name in zip(df_nut_norm["id"], df_nut_norm.get("name", pd.Series([None]*len(df_nut_norm))))]
        sel_n = st.selectbox("Choose a recipe (in the sample)", options_n, key="nutri_select")
        selected_id_n = int(str(sel_n).split(" — ")[0])

        st.subheader("Selected Recipe")
        sel_row_n = df_nut_norm[df_nut_norm["id"] == selected_id_n]
        if not sel_row_n.empty:
            r = sel_row_n.iloc[0]
            st.markdown(f"**{r.get('name','—')}**  \nminutes : **{r.get('minutes','—')}**, steps : **{r.get('n_steps','—')}**")
            nutri_cols = [c for c in cn.NUTRI_COLS if c in df_nut_norm.columns]
            if nutri_cols:
                st.write({c: r.get(c, '—') for c in nutri_cols})

        with st.spinner("Calculating nutritional similarity..."):
            try:
                neighbours_n = cn.find_similar_by_nutri(sim_nut, selected_id_n, top_n=nutri_top_n)
            except KeyError:
                st.error("Selected recipe not found in the nutritional similarity matrix.")
                neighbours_n = pd.Series(dtype=float)

        if neighbours_n.empty:
            st.info("No similar recipes found for this selection.")
        else:
            neigh_df_n = neighbours_n.reset_index().rename(columns={"index": "id", selected_id_n: "score"})
            neigh_df_n = neigh_df_n.merge(df_nut_norm, on="id", how="left")
            display_cols_n = [c for c in ["id", "name", "minutes", "n_steps"] + nutri_cols if c in neigh_df_n.columns]
            neigh_df_n = neigh_df_n[["id", "score"] + [c for c in display_cols_n if c not in ("id")]]
            st.subheader(f"Top {len(neigh_df_n)} similar recipes (nutrients)")
            st.dataframe(neigh_df_n.fillna("—"))

            if not neigh_df_n.empty:
                inspect_id_n = st.selectbox("Inspect similar recipe (nutrients)", neigh_df_n["id"].astype(str), key="nutri_inspect")
                insp_row_n = neigh_df_n[neigh_df_n["id"].astype(str) == str(inspect_id_n)].iloc[0]
                st.markdown(f"### {insp_row_n.get('name','—')}")
                st.write({k: insp_row_n.get(k) for k in ["minutes", "n_steps"] + nutri_cols if k in insp_row_n.index})


if __name__ == "__main__":
    app()
