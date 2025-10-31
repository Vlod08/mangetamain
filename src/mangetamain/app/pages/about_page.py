# app/pages/07_About.py
from __future__ import annotations
import streamlit as st
from app.app_utils.ui import use_global_ui

use_global_ui(
    page_title="Mangetamain — About",
    subtitle="Big Data Project - Télécom Paris",
    logo="assets/mangetamain-logo.jpg",
    logo_size_px=90,
    round_logo=True,
)

# =========================
# Config
# =========================
REPO_URL   = "https://github.com/Vlod08/mangetamain"          
DOCS_URL   = "https://vlod08.github.io/mangetamain/"                    
KAGGLE_URL = "https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions"

TEAM = [
    {
        "name": "Mohammed Khalil OUNIS",
        "email": "khalilounis10@gmail.com",
        "github": None,
        "linkedin": None,
    },
    {
        "name": "Bryan LY",
        "email": "bryan29.ly@gmail.com",
        "github": None,
        "linkedin": None,
    },
    {
        "name": "Lina RHIATI HAZIME",
        "email": "lina.rhiati2@gmail.com",
        "github": None,
        "linkedin": None,
    },
    {
        "name": "Mohammed ELAMINE",
        "email": "elamine.mohammed.14@gmail.com",
        "github": None,
        "linkedin": None,
    },
    {
        "name": "Lokeshwaran VENGADABADY",
        "email": "lokeshvengadabady@gmail.com",
        "github": None,
        "linkedin": None,
    },
]

# =========================
# About
# =========================
st.markdown("## 🧾 About the project")
st.write(
    f"""
**Mangetamain** is an interactive web application developed as part of the **Big Data Project (Kit Big Data)** at **Télécom Paris**.  
It allows users to explore recipes from [Food.com]({KAGGLE_URL}) and perform various analyses:
- **Culinary trends** (time, ingredients, seasons, countries, etc.)
- **User behavior** (ratings, reviews, biases, text analysis)
- **Data quality** and **visual exploration** using Streamlit and Plotly
- **Project administration** (logs, consistency, statistics)
"""
)

# =========================
# Team section
# =========================
st.divider()
st.markdown("## 👩‍💻 Project Team")

cols = st.columns(2)
for i, m in enumerate(TEAM):
    with cols[i % 2]:
        links = []
        if m.get("github"):
            links.append(f"🔗 [GitHub]({m['github']})")
        if m.get("linkedin"):
            links.append(f"🔗 [LinkedIn]({m['linkedin']})")
        links_txt = " · ".join(links) if links else ""

        st.markdown(
            f"""
**{m['name']}**  
📧 [{m['email']}](mailto:{m['email']})  
{links_txt}
"""
        )

# =========================
# Resources & links
# =========================
st.divider()
st.markdown("## 🔗 Resources & Useful Links")
st.markdown(
    f"""
- 📂 **GitHub Repository**: [{REPO_URL}]({REPO_URL})
- 📘 **Sphinx Documentation**: [{DOCS_URL}]({DOCS_URL})
- 🗂️ **Kaggle Dataset**: [Food.com Recipes & Interactions]({KAGGLE_URL})
- 🧩 **Main Technologies**:
  - Python 3.10+
  - Streamlit, Plotly, Seaborn, Matplotlib
  - pandas, scikit-learn, NumPy
  - Logging, Sphinx, Poetry
"""
)

# =========================
# Project architecture
# =========================
st.divider()
st.markdown("## 🧱 Project Architecture")
st.code(
    """\
src/
├── app/                # Streamlit interface (pages, UI, utils)
│   ├── pages/          # Multipage Streamlit pages
│   └── app_utils/      # UI / IO / visualization helpers
├── core/               # Business logic & services (datasets, analysis)
│   ├── dataset.py      # Data access (recipes, reviews)
│   ├── recipes_service.py
│   └── reviews_service.py
├── data/               # Raw data and preprocessed artifacts
├── docs/               # Sphinx documentation
├── logs/               # Application logs
└── tests/              # Unit tests and checks
""",
    language="text",
)

# (Optional) footer
st.caption("© 2025 — Mangetamain Team · Télécom Paris")
