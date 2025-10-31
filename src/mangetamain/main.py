# main.py
from __future__ import annotations
import streamlit as st
from streamlit_lottie import st_lottie
import time

from mangetamain.core.utils.utils import setup_logging
from mangetamain.config import ROOT_DIR
from mangetamain.core.dataset import DatasetLoaderThread, RecipesDataset, InteractionsDataset
from mangetamain.core.utils.utils import load_lottie
import os
import zipfile
import requests


def download_and_unzip_kaggle_dataset(
    url: str = "https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions",
    download_dir: str | None = None,
):
    """
    Download and unzip the Kaggle dataset into download_dir. If download_dir is None,
    it defaults to the repository data folder (ROOT_DIR / 'data').
    """
    # default to repository data folder if not provided
    if download_dir is None:
        download_dir = str(ROOT_DIR / "data")

    download_dir = os.path.expanduser(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    zip_path = os.path.join(download_dir, "food-com-recipes-and-user-interactions.zip")

    # Use requests to stream the download. Note: Kaggle API usually requires credentials.
    print("Downloading dataset...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Download complete: {zip_path}")
    print("Unzipping contents...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    print(f"Files extracted to: {download_dir}")

def pages():

    # --- Pages ---
    pages_dir = ROOT_DIR / "src" / "mangetamain" / "app" / "pages"
    home_page = st.Page(
        pages_dir / "home_page.py", 
        title="Home", 
        icon="🏠", 
        url_path="",
        default=True
    )
    explorer_recipes_page = st.Page(
        pages_dir / "recipes" / "recipes_explorer_page.py", 
        title="Recipes Explorer", 
        url_path="/recipes",
        icon="📖"
    )
    explorer_interactions_page = st.Page(
        pages_dir / "interactions" / "interactions_explorer_page.py", 
        title="Interactions Explorer", 
        url_path="/interactions",
        icon="🔍"
    )
    country_season_page = st.Page(
        pages_dir / "recipes" / "country_season_page.py", 
        title="Country & Season", 
        url_path="/recipes_country_season",
        icon="🌍"
    )
    clustering_page = st.Page(
        pages_dir / "recipes" / "clustering_page.py", 
        title="Clustering", 
        url_path="/recipes_clustering",
        icon="🧑‍🤝‍🧑"
    )
    interactions_analysis_page = st.Page(
        pages_dir / "interactions" / "interactions_analysis_page.py", 
        title="Interactions Analysis", 
        url_path="/interactions_analysis",
        icon="📊"
    )
    admin_page = st.Page(
        pages_dir / "admin_page.py", 
        title="Admin", 
        url_path="/admin",
        icon="🛠️"
    )
    about_page = st.Page(
        pages_dir / "about_page.py", 
        title="About", 
        url_path="/about",
        icon="ℹ️"
    )
    pg = st.navigation(
        {
            "Main Menu": [home_page, admin_page, about_page],
            "Recipes": [explorer_recipes_page, country_season_page, clustering_page],
            "Interactions": [explorer_interactions_page, interactions_analysis_page],
        }, 
        position="top",
        expanded=False
    )
    return pg

def load_data(name: str = None) -> None:
    """Simulate loading and preprocessing with progress."""
    
    if name is None:
        st.markdown("#### Recipes & Interactions Data Loading")
    elif "recipe" in name.lower():
        st.markdown("#### Recipes Data Loading")
    elif "interaction" in name.lower():
        st.markdown("#### Interactions Data Loading")
    else:
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
    text_placeholder.text("Data loading...")
    progress_bar = progress_placeholder.progress(0)

    # Use threads to load datasets concurrently
    threads = []
    if (name is None) or (name == "recipes"):
        threads.append(DatasetLoaderThread(
            RecipesDataset(), 
            label="recipes"
            ))
    if (name is None) or (name == "interactions"):
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
    
    if name is None:
        st.session_state["data_ready"] = True

    # Cool success animation
    # st_lottie(load_lottie("success_check.json"), height=150, speed=0.7, loop=False)
    if name is None:
        st.success("All datasets loaded successfully!")
    elif "recipe" in name.lower():
        st.success("Recipes dataset loaded successfully!")
    elif "interaction" in name.lower():
        st.success("Interactions dataset loaded successfully!")
    else:
        st.success("All datasets loaded successfully!")
    st.balloons()
    time.sleep(2)


def app():
    # Ensure dataset exists locally; attempt download from Kaggle if missing.
    data_dir = ROOT_DIR / "data"
    recipes_parquet = data_dir / "processed" / "recipes.parquet"
    recipes_csv = data_dir / "RAW_recipes.csv"
    interactions_parquet = data_dir / "processed" / "interactions.parquet"
    interactions_csv = data_dir / "RAW_interactions.csv"
    # If either recipes or interactions datasets are missing, attempt automatic download
    if not (recipes_parquet.exists() or recipes_csv.exists()) or not (interactions_parquet.exists() or interactions_csv.exists()):
        try:
            download_and_unzip_kaggle_dataset(download_dir=str(data_dir))
            st.info("Dataset download attempted; continuing to load the app.")
        except Exception as e:
            st.warning(f"Automatic dataset download failed: {e}")

    pg = pages()
    # if both datasets are already loaded, then just run the page
    if "data_ready" in st.session_state and st.session_state["data_ready"]:
        pg.run()
        return
    # Only load the dataset we want in case we visit a specific group page
    if "recipes" in pg.url_path.lower():
        if ("recipes" not in st.session_state) \
            or (st.session_state["recipes"] is None) \
                or (st.session_state["recipes"].empty):
            load_data("recipes")
    elif "interactions" in pg.url_path.lower():
        if ("interactions" not in st.session_state) \
            or (st.session_state["interactions"] is None) \
                or (st.session_state["interactions"].empty):
            load_data("interactions")
    # otherwise, load both datasets
    else:
        load_data()
    pg.run()

if __name__ == "__main__":
    setup_logging()
    app()
