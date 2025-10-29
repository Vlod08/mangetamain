# app/pages/home_page.py
from __future__ import annotations
import time
import streamlit as st
# from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit_lottie import st_lottie
from app.app_utils.ui import use_global_ui
from core.dataset import DatasetLoaderThread, RecipesDataset, InteractionsDataset
from config import ROOT_DIR
import json


@st.cache_data(show_spinner=False)
def load_lottie():
    filepath = ROOT_DIR / "assets" / "prepare_food.json"
    with open(filepath, "r") as f:
        return json.load(f)

def load_data():
    """Simulate loading and preprocessing with progress."""
    
    st.markdown("#### Recipes & Interactions Data Loading")

    # Placeholders
    lottie_placeholder = st.empty()
    progress_placeholder = st.empty()
    text_placeholder = st.empty()

    # Load and display Lottie animation
    lottie = load_lottie()
    with lottie_placeholder:
        st_lottie(lottie, height=200, speed=1, loop=True)
    
    # Progress bar and text
    text_placeholder.text("Starting data loading...")
    progress_bar = progress_placeholder.progress(0)

    # Use threads to load datasets concurrently
    threads = []
    threads.append(DatasetLoaderThread(
        RecipesDataset(), 
        label="recipes"
        ))
    threads.append(DatasetLoaderThread(
        InteractionsDataset(), 
        label="interactions"
    ))

    # Start threads
    for thread in threads:
        thread.start()

    # Monitor progress
    thread_lives = [True] * len(threads)
    while any(thread_lives):
        for i, thread in enumerate(threads):
            if thread_lives[i] and not thread.is_alive():
                st.session_state[thread.label] = thread.return_value
                thread_lives[i] = False
                progress_bar.progress(100 - sum(thread_lives) * 50)
                text_placeholder.text(f"{thread.label.capitalize()} dataset loaded.")
        time.sleep(0.5)    

    # Ensure all threads have finished
    for thread in threads:
        thread.join()

    # --- Replace loading elements with success message ---
    lottie_placeholder.empty()
    progress_placeholder.empty()
    text_placeholder.empty()

    st.toast("Data is now ready!", icon=":material/thumb_up:", duration=5)
    st.session_state["data_ready"] = True

    # Cool success animation
    # st_lottie(load_lottie("success_check.json"), height=150, speed=0.7, loop=False)
    st.success("All datasets loaded successfully!")
    st.balloons()

def app():
    # --- UI
    use_global_ui(
        page_title="Mangetamain App",
        subtitle="Explore and analyze the Food.com recipes dataset",
        logo="assets/mangetamain-logo.jpg", 
        logo_size_px=90,
        round_logo=True,
    )

    # --- Main content ---
    st.markdown("### Welcome to the Mangetamain App!")
    st.markdown("Explore and analyze the Food.com recipes dataset.")
    st.markdown("Use the navigation menu to access different sections of the app.")

    # ======== Data Loading =========
    # Preload EDA services for recipes & interactions
    # Trigger loading in background (async-like behavior)

    # Initialize data_ready flag
    if "data_ready" not in st.session_state:
        st.session_state["data_ready"] = False

    if not st.session_state["data_ready"]:
        load_data()
    else:
        st.success("Data already loaded! 🎉")
        st.balloons()

if __name__ == "__main__":
    app()