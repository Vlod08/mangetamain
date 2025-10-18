# app_utils/ui.py
import streamlit as st
from textwrap import dedent
import base64, os, mimetypes

def _data_url_from_path(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"  
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def use_global_ui(page_title: str = "Mangetamain",
                  page_icon: str | None = None,
                  subtitle: str | None = None,
                  wide: bool = True,
                  logo: str | None = None,        
                  logo_size_px: int = 56,         
                  round_logo: bool = True):      
    """Applique la config de page + CSS global + header cohérent."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon if page_icon else "🍲",
        layout="wide" if wide else "centered",
        initial_sidebar_state="expanded",
    )

    # ---- CSS Global (affine la UI native Streamlit) ----
    st.markdown(dedent(f"""
    <style>
      /* Conteneur général : limite la largeur utile pour éviter les lignes trop longues */
      .block-container {{ max-width: 1200px; padding-top: 1.2rem; padding-bottom: 4rem; }}

      /* Boutons */
      .stButton > button {{
        border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600;
        border: 1px solid rgba(0,0,0,0.05);
      }}

      /* Selects, inputs */
      .stSelectbox, .stTextInput, .stNumberInput, .stMultiSelect, .stDateInput {{
        border-radius: 10px;
      }}

      /* Tabs plus visibles */
      .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; }}
      .stTabs [data-baseweb="tab"] {{
        border-radius: 10px; padding: 0.4rem 0.8rem;
      }}

      /* Cartes de métriques : taille + alignement */
      [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
      [data-testid="stMetricDelta"] {{ font-size: 0.9rem; }}

      /* Tables (AgGrid/df) : arrondis légers */
      .stDataFrame, .stTable {{ border-radius: 10px; overflow: hidden; }}

      /* Footer & burger menu si tu veux une UI épurée */
      footer {{visibility: hidden;}}
      #MainMenu {{visibility: hidden;}}

      /* ✅ Style de l'image/logo du header */
      .mtm-logo {{
        width: {logo_size_px}px; height: {logo_size_px}px; object-fit: cover;
        {"border-radius: 50%;" if round_logo else ""}
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
      }}
    </style>
    """), unsafe_allow_html=True)

    # ---- Résolution du logo (fichier local -> data URL ; URL directe inchangée) ----
    logo_src = None
    if logo:
        if logo.startswith("http://") or logo.startswith("https://") or logo.startswith("data:"):
            logo_src = logo
        elif os.path.exists(logo):
            logo_src = _data_url_from_path(logo)

    # ---- Header cohérent ----
    left, right = st.columns([1, 5], vertical_alignment="center")
    with left:
        if logo_src:
            st.markdown(f'<img class="mtm-logo" src="{logo_src}" alt="logo">', unsafe_allow_html=True)
        else:
            st.markdown("### 🍲")  # fallback si aucun logo fourni
    with right:
        st.markdown(f"## **{page_title}**")
        if subtitle:
            st.markdown(subtitle)

    st.divider()
