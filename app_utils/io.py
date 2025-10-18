from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import streamlit as st

# 1) On PRÉFÈRE l'artefact propre, puis on retombe sur le RAW si besoin
DATA_PATHS = ( # artefact propre
    "data/recipes_clean.parquet",
    "data/RAW_recipes.csv",                  # brut
    "./RAW_recipes.csv",
    "RAW_recipes.csv",
)

# 2) Contrat minimal attendu par l'app (lecture seule)
REQUIRED_COLS = {
    "id", "name", "minutes", "n_steps", "n_ingredients",
    "tags", "ingredients", "description", "submitted"
}
# Colonnes nutrition optionnelles (si présentes, tant mieux)
OPTIONAL_COLS = {
    "calories","total_fat","sugar","sodium","protein","saturated_fat","carbohydrates",
    "nutrition_cal","nutrition_fat","nutrition_sugar","nutrition_protein","nutrition_sodium"
}

def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # 1) Cherche le premier fichier dispo
    for p in DATA_PATHS:
        if os.path.exists(p):
            df = _read_any(p)
            break
    else:
        raise FileNotFoundError(
            "Aucune donnée trouvée.\n"
            "Attendu : data/processed/recipes_clean.parquet OU data/RAW_recipes.csv"
        )

    # 2) Assure colonnes minimales (ajoute vides si manquent pour éviter les crashs d'affichage)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = None

    # 3) Types doux
    for c in ("minutes","n_steps","n_ingredients"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["submitted"].notna().any():
        df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")

    return df

def validate_schema(df: pd.DataFrame) -> dict:
    """Retourne un petit rapport de validité du schéma pour la page qualité."""
    missing = sorted([c for c in REQUIRED_COLS if c not in df.columns])
    optional_present = sorted([c for c in OPTIONAL_COLS if c in df.columns])
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "optional_present": optional_present,
        "rows": len(df),
        "cols": df.shape[1],
    }

def artifact_path() -> Path:
    """Chemin de l'artefact propre conseillé."""
    return Path("data/processed/recipes_clean.parquet")
