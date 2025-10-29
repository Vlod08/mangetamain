# main.py
from __future__ import annotations
import streamlit as st

from core.utils.utils import setup_logging
from config import ROOT_DIR

def app():

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
    pg.run()

if __name__ == "__main__":
    setup_logging()
    app()
