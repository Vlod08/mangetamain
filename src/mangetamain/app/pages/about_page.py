# app/pages/about_page.py
from __future__ import annotations
import streamlit as st

from app.app_utils.ui import use_global_ui


def app():
    use_global_ui(
        "Mangetamain — About",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
        subtitle=None,
        wide=True,
    )

    st.markdown(
        """
    **Mangetamain** — demonstration application.  
    This app covers: quality, exploration, countries, seasons/events, clustering, admin.
    - Dataset: Food.com (Kaggle)
    - Tech: Streamlit, Plotly, scikit-learn
    - Authors: Team Mangetamain
    """
    )


if __name__ == "__main__":
    app()
