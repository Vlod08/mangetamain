# main.py
from __future__ import annotations
import streamlit as st
from streamlit_lottie import st_lottie
import time
from pathlib import Path

# from mangetamain.core.utils.utils import setup_logging
from mangetamain.config import (
    ROOT_DIR, DATA_DIR,
    RECIPES_CSV, RECIPES_PARQUET,
    INTERACTIONS_CSV, INTERACTIONS_PARQUET,
    DB_NAME
)
from mangetamain.core.dataset import (
    DatasetLoaderThread,
    RecipesDataset,
    InteractionsDataset,
)
from mangetamain.core.utils.utils import load_lottie
from mangetamain.core.app_logging import setup_logging, get_logger
import os
import zipfile
import requests


def download_and_unzip_kaggle_dataset(
    url: str = "https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions",
    download_dir: Path | None = None,
    target_files: list[str] | None = None,
):
    """
    Download and unzip specific files from the Kaggle dataset into download_dir.
    If download_dir is None, defaults to the repository data folder (DATA_DIR).
    """
    if download_dir is None:
        download_dir = str(DATA_DIR)
    elif isinstance(download_dir, Path):
        download_dir = str(download_dir)

    download_dir = os.path.expanduser(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    zip_path = os.path.join(download_dir, "food-com-recipes-and-user-interactions.zip")

    # Default files to extract if none are specified
    if target_files is None:
        target_files = ["RAW_recipes.csv", "RAW_interactions.csv"]

    st.session_state["logger"].info("Downloading dataset...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    st.session_state["logger"].info(f"Download complete: {zip_path}")
    st.session_state["logger"].info("Extracting selected files...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        all_files = zip_ref.namelist()
        for file in target_files:
            matched_files = [f for f in all_files if f.endswith(file)]
            if matched_files:
                for f in matched_files:
                    zip_ref.extract(f, download_dir)
                    st.session_state["logger"].info(f"Extracted: {f}")
            else:
                st.session_state["logger"].warning(f"File not found in archive: {file}")

    # Safely remove the zip file after extraction
    try:
        os.remove(zip_path)
        st.session_state["logger"].info(f"Removed zip file: {zip_path}")
    except OSError as e:
        st.session_state["logger"].warning(f"Could not remove zip file: {e}")

    st.session_state["logger"].info(f"Selected files extracted to: {download_dir}")
    return Path(download_dir)


def pages() -> st.StreamlitPage:
    """Define the application pages."""
    # --- Pages ---
    pages_dir = ROOT_DIR / "src" / "mangetamain" / "app" / "pages"
    home_page = st.Page(
        pages_dir / "home_page.py", title="Home", icon="🏠", url_path="", default=True
    )
    explorer_recipes_page = st.Page(
        pages_dir / "recipes" / "recipes_explorer_page.py",
        title="Recipes Explorer",
        url_path="/recipes",
        icon="📖",
    )
    explorer_interactions_page = st.Page(
        pages_dir / "interactions" / "interactions_explorer_page.py",
        title="Interactions Explorer",
        url_path="/interactions",
        icon="🔍",
    )
    country_season_page = st.Page(
        pages_dir / "recipes" / "country_season_page.py",
        title="Country & Season",
        url_path="/recipes_country_season",
        icon="🌍",
    )
    clustering_page = st.Page(
        pages_dir / "recipes" / "clustering_page.py",
        title="Clustering",
        url_path="/recipes_clustering",
        icon="🧑‍🤝‍🧑",
    )
    interactions_analysis_page = st.Page(
        pages_dir / "interactions" / "interactions_analysis_page.py",
        title="Interactions Analysis",
        url_path="/interactions_analysis",
        icon="📊",
    )
    admin_page = st.Page(
        pages_dir / "admin_page.py", title="Admin", url_path="/admin", icon="🛠️"
    )
    about_page = st.Page(
        pages_dir / "about_page.py", title="About", url_path="/about", icon="ℹ️"
    )
    pg = st.navigation(
        {
            "Main Menu": [home_page, admin_page, about_page],
            "Recipes": [explorer_recipes_page, country_season_page, clustering_page],
            "Interactions": [explorer_interactions_page, interactions_analysis_page],
        },
        position="top",
        expanded=False,
    )
    return pg


def load_data(name: str = None, data_dir: str = None) -> None:
    """Simulate loading and preprocessing with progress.
    If name is None, load both datasets. Otherwise, load only the specified dataset.
    """

    # Placeholders
    header_placeholder = st.empty()
    lottie_placeholder = st.empty()
    progress_placeholder = st.empty()
    text_placeholder = st.empty()

    if name is None:
        header_placeholder.header("Recipes & Interactions Data Loading")
    elif "recipe" in name.lower():
        header_placeholder.header("Recipes Data Loading")
    elif "interaction" in name.lower():
        header_placeholder.header("Interactions Data Loading")
    else:
        header_placeholder.header("Recipes & Interactions Data Loading")

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
        threads.append(DatasetLoaderThread(RecipesDataset(data_dir=data_dir), label="recipes"))
    if (name is None) or (name == "interactions"):
        threads.append(DatasetLoaderThread(InteractionsDataset(data_dir=data_dir), label="interactions"))

    # Start threads
    for thread in threads:
        thread.start()

    # Monitor progress
    thread_lives = [True] * len(threads)
    while any(thread_lives):
        for i, thread in enumerate(threads):
            if thread_lives[i] and not thread.is_alive():
                st.session_state[thread.label] = thread.return_value
                if "issues" not in st.session_state:
                    st.session_state["issues"] = {}
                st.session_state["issues"][thread.label] = thread.issues
                thread_lives[i] = False
                progress_bar.progress(100 - sum(thread_lives) * 50)
                text_placeholder.text(f"{thread.label.capitalize()} dataset loaded.")
        time.sleep(0.5)

    # Ensure all threads have finished
    for thread in threads:
        thread.join()

    # --- Replace loading elements with success message ---
    header_placeholder.empty()
    lottie_placeholder.empty()
    progress_placeholder.empty()
    text_placeholder.empty()

    st.toast("Data is now ready!", icon=":material/thumb_up:", duration=5)

    if name is None:
        st.session_state["data_ready"] = True

    # Cool success animation
    # st_lottie(load_lottie("success_check.json"), height=150, speed=0.7, loop=False)
    success_placeholder = st.empty()
    if name is None:
        with success_placeholder:
            st.success("All datasets loaded successfully!")
    elif "recipe" in name.lower():
        with success_placeholder:
            st.success("Recipes dataset loaded successfully!")
    elif "interaction" in name.lower():
        with success_placeholder:
            st.success("Interactions dataset loaded successfully!")
    else:
        with success_placeholder:
            st.success("All datasets loaded successfully!")
    st.balloons()
    time.sleep(2)
    success_placeholder.empty()


def app():
    # Ensure dataset exists locally; attempt download from Kaggle if missing.
    recipes_parquet = DATA_DIR / RECIPES_PARQUET
    recipes_csv = DATA_DIR / RECIPES_CSV
    interactions_parquet = DATA_DIR / INTERACTIONS_PARQUET
    interactions_csv = DATA_DIR / INTERACTIONS_CSV
    db_path = DATA_DIR / DB_NAME

    pg = pages()

    # if both datasets are already loaded, then just run the page
    if "data_ready" in st.session_state and st.session_state["data_ready"]:
        pg.run()
        return

    # If either recipes or interactions datasets are missing, attempt automatic download
    recipes_dataset_missing = True
    if recipes_parquet.exists() or recipes_csv.exists():
        st.session_state["logger"].info("Recipes dataset found locally.")
        recipes_dataset_missing = False
    else:
        st.session_state["logger"].warning("Recipes dataset not found locally.")

    interactions_dataset_missing = True
    if interactions_parquet.exists() or interactions_csv.exists():
        st.session_state["logger"].info("Interactions dataset found locally.")
        interactions_dataset_missing = False
    else:
        st.session_state["logger"].warning("Interactions dataset not found locally.")

    if recipes_dataset_missing or interactions_dataset_missing:# or (not db_path.exists()):
        try:
            st.write("Datasets not found locally. Attempting to download from Kaggle...")
            download_dir = download_and_unzip_kaggle_dataset(download_dir=str(DATA_DIR))
            st.success("Datasets downloaded and extracted successfully!")
        except Exception as e:
            st.warning(f"Automatic dataset download failed: {e}")
    else:
        download_dir = DATA_DIR
    
    # Only load the dataset we want in case we visit a specific group page
    if "recipes" in pg.url_path.lower():
        if (
            ("recipes" not in st.session_state)
            or (st.session_state["recipes"] is None)
            or (st.session_state["recipes"].empty)
        ):
            load_data("recipes", data_dir=download_dir)
    elif "interactions" in pg.url_path.lower():
        if (
            ("interactions" not in st.session_state)
            or (st.session_state["interactions"] is None)
            or (st.session_state["interactions"].empty)
        ):
            load_data("interactions", data_dir=download_dir)
    # otherwise, load both datasets
    else:
        recipes_loaded = True
        interactions_loaded = True
        if (
            "recipes" not in st.session_state
            or (st.session_state["recipes"] is None)
            or (st.session_state["recipes"].empty)
        ):
            recipes_loaded = False
        if (
            "interactions" not in st.session_state
            or (st.session_state["interactions"] is None)
            or (st.session_state["interactions"].empty)
        ):
            interactions_loaded = False

        placeholders = st.empty()
        if (not recipes_loaded) and (not interactions_loaded):
            load_data(data_dir=download_dir)
        elif not recipes_loaded:
            with placeholders:
                st.success("Interactions dataset already loaded in memory!")
            load_data("recipes", data_dir=download_dir)
        elif not interactions_loaded:
            with placeholders:
                st.success("Recipes dataset already loaded in memory!")
            load_data("interactions", data_dir=download_dir)
        placeholders.empty()
    pg.run()


if __name__ == "__main__":
    if "logger" not in st.session_state:
        setup_logging()
        st.session_state["logger"] = get_logger("mangetamain.main")
    else:
        st.session_state["logger"].info("-" * 50)
        st.session_state["logger"].info("Application started/restarted.")
    app()
