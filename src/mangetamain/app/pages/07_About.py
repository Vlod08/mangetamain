from pathlib import Path
import sys
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app_utils.ui import use_global_ui

use_global_ui(
    "Mangetamain — À propos",
    logo="image/image.jpg",
    logo_size_px=90,
    round_logo=True,
    subtitle=None,
    wide=True,
)

# st.title("ℹ️ À propos")
st.markdown(
    """
**Mangetamain** — application de démonstration.  
Cette app couvre : qualité, exploration, pays, saisons/événements, clustering, admin.
- Dataset : Food.com (Kaggle)
- Tech : Streamlit, Plotly, scikit-learn
- Auteurs : Équipe Mangetamain
"""
)
