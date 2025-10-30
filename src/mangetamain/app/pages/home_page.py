# app/pages/home_page.py
from __future__ import annotations
import streamlit as st
# from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from app.app_utils.ui import use_global_ui


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

if __name__ == "__main__":
    app()