# src/mangetamain/app/pages/home_page.py
from __future__ import annotations

import streamlit as st

# from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from app.app_utils.ui import use_global_ui


import json
import time

from streamlit_lottie import st_lottie

from core.dataset import (
    DatasetLoaderThread,
    RecipesDataset,
    InteractionsDataset,
)
from core.app_logging import get_logger
from config import ROOT_DIR


LOGGER = get_logger()


@st.cache_data(show_spinner=False)
def load_lottie() -> dict:
    filepath = ROOT_DIR / "assets" / "prepare_food.json"
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _render_project_intro() -> None:
    """Intro block shown after / with the loader."""
    st.markdown("## Project context")
    st.write(
        """
**Mangetamain** is a data-science project (Télécom Paris — Big Data kit) built on the
public **Food.com** dataset. It lets you explore recipes and more than **1M** user
interactions (ratings, comments, dates).
"""
    )

    st.markdown("## Goals")
    st.write(
        """
1. **Clean and structure** two large datasets:
   - 🥘 *Recipes*: ~230k recipes
   - 💬 *Interactions / reviews*: >1M rows
2. **Explore & visualize** culinary features (ingredients, time, tags, nutrition)
3. **Analyze user behavior** (rating distribution, review length, user bias)
4. **Produce fast, preprocessed artefacts** (Parquet) that the Streamlit app can load quickly
"""
    )

    st.markdown("## Methodology & tools")
    st.write(
        """
- Modular Python architecture
- `core/` → data services (`RecipesDataset`, `InteractionsDataset`)
- `app/pages/` → Streamlit multipage UI
- Main libs: **pandas, numpy, plotly, matplotlib, seaborn, streamlit**
"""
    )

    st.markdown("## Expected outcomes")
    st.write(
        """
- 🔍 An intuitive exploration of the Food.com data  
- 📈 Interactive visualizations on culinary trends  
- 🧠 Advanced analysis leads (clustering, user profiles, temporal patterns)  
- 🍴 A better understanding of user preferences by **season**, **country**, or **cooking time**
"""
    )

    st.markdown("## Final vision")
    st.write(
        """
**Mangetamain** is meant to be a small data-product for food data.  
The idea is to show that even “everyday” data like recipes and reviews can reveal
**cultural, seasonal and behavioral patterns** when you clean it and visualize it properly.

> “Behind every recipe, there’s a story of taste, time and sharing.”
"""
    )

    st.divider()

    st.markdown("### Where to go next?")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Use the **Recipes** menu (top) to explore the recipe dataset.")
    with col2:
        st.info("Use the **Interactions** menu (top) to explore reviews / ratings.")


def _load_datasets_once() -> None:
    """
    Load recipes + interactions in parallel and store them in session_state.
    If already loaded, do nothing.
    """
    if st.session_state.get("data_ready", False):
        LOGGER.info("Home: datasets already in session_state -> skip load")
        return

    LOGGER.info("Home: starting dataset loading…")
    st.markdown("#### Loading recipes & interactions datasets")

    # placeholders
    lottie_placeholder = st.empty()
    progress_placeholder = st.empty()
    text_placeholder = st.empty()

    # animation
    lottie = load_lottie()
    with lottie_placeholder:
        st_lottie(lottie, height=200, speed=1, loop=True)

    text_placeholder.text("Starting data loading…")
    progress_bar = progress_placeholder.progress(0)

    # run both loaders in parallel
    threads = [
        DatasetLoaderThread(RecipesDataset(), label="recipes"),
        DatasetLoaderThread(InteractionsDataset(), label="interactions"),
    ]

    for t in threads:
        t.start()

    thread_alive = [True] * len(threads)
    step = 100 // len(threads)  # 50/50 here
    while any(thread_alive):
        for i, t in enumerate(threads):
            if thread_alive[i] and not t.is_alive():
                # store result
                st.session_state[t.label] = t.return_value
                thread_alive[i] = False
                done = (len(threads) - sum(thread_alive)) * step
                progress_bar.progress(min(done, 100))
                text_placeholder.text(f"{t.label.capitalize()} dataset loaded.")
        time.sleep(0.4)

    for t in threads:
        t.join()

    # clean loading UI
    lottie_placeholder.empty()
    progress_placeholder.empty()
    text_placeholder.empty()

    st.toast("Data is now ready!", icon=":material/thumb_up:")
    st.success("All datasets loaded successfully!")
    st.session_state["data_ready"] = True
    LOGGER.info("Home: datasets loaded and stored in session_state")


def app() -> None:
    # top banner

    use_global_ui(
        page_title="Mangetamain — Data & Cuisine",
        subtitle="Explore and analyze the Food.com recipes dataset",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
    )

    # --- Main content ---
    st.markdown("### Welcome to the Mangetamain App!")
    st.markdown("Explore and analyze the Food.com recipes dataset.")
    st.markdown("Use the navigation menu to access different sections of the app.")

    # make sure key exists
    st.session_state.setdefault("data_ready", False)

    if not st.session_state["data_ready"]:
        # first time → show loader, then intro
        _load_datasets_once()
        _render_project_intro()
    else:
        # already loaded → small banner + intro
        st.success("Data already loaded! 🎉")
        _render_project_intro()


if __name__ == "__main__":
    app()
