# app_utils/ui.py
import streamlit as st
from textwrap import dedent

def use_global_ui(page_title: str = "Mangetamain",
                  page_icon: str = "🍲",
                  subtitle: str | None = None,
                  wide: bool = True):
    """Applique la config de page + CSS global + header cohérent."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide" if wide else "centered",
        initial_sidebar_state="expanded",
    )

    # ---- CSS Global (affine la UI native Streamlit) ----
    st.markdown(dedent("""
    <style>
      /* Conteneur général : limite la largeur utile pour éviter les lignes trop longues */
      .block-container { max-width: 1200px; padding-top: 1.2rem; padding-bottom: 4rem; }

      /* Boutons */
      .stButton > button {
        border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600;
        border: 1px solid rgba(0,0,0,0.05);
      }

      /* Selects, inputs */
      .stSelectbox, .stTextInput, .stNumberInput, .stMultiSelect, .stDateInput {
        border-radius: 10px;
      }

      /* Tabs plus visibles */
      .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
      .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 0.4rem 0.8rem;
      }

      /* Cartes de métriques : taille + alignement */
      [data-testid="stMetricValue"] { font-size: 1.6rem; }
      [data-testid="stMetricDelta"] { font-size: 0.9rem; }

      /* Tables (AgGrid/df) : arrondis légers */
      .stDataFrame, .stTable { border-radius: 10px; overflow: hidden; }

      /* Footer & burger menu si tu veux une UI épurée */
      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}
    </style>
    """), unsafe_allow_html=True)

    # ---- Header cohérent ----
    left, right = st.columns([1, 5], vertical_alignment="center")
    with left:
        st.markdown("### 🍲")
    with right:
        st.markdown(f"## **{page_title}**")
        if subtitle:
            st.markdown(subtitle)

    st.divider()
