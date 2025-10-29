# app/app_utils/io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# --- Trouver la racine du repo peu importe le dossier courant ---
def find_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data").exists():
            return cand
    return Path.cwd()

ROOT = find_root()

# 1) On PRÉFÈRE l'artefact propre, puis on retombe sur le RAW si besoin
DATA_PATHS = (
    ROOT / "data" / "processed" / "recipes_clean.parquet",  # artefact propre
    ROOT / "data" / "raw" / "RAW_recipes.csv",              # brut
    ROOT / "RAW_recipes.csv",                               # fallback (ancien)
)

# 2) Contrat minimal attendu par l'app (lecture seule)
REQUIRED_COLS = {
    "id", "name", "minutes", "n_steps", "n_ingredients",
    "tags", "ingredients", "description", "submitted"
}

OPTIONAL_COLS = {
    "calories","total_fat","sugar","sodium","protein",
    "saturated_fat","carbohydrates",
    "nutrition_cal","nutrition_fat","nutrition_sugar",
    "nutrition_protein","nutrition_sodium"
}

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)  # needs pyarrow or fastparquet
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # 1) Look for the first available data file
    df = None
    for p in DATA_PATHS:
        if p.exists():
            df = _read_any(p)
            break
    if df is None:
        raise FileNotFoundError(
            f"Aucune donnée trouvée.\n"
            f"Attendu : {DATA_PATHS[0]} OU {DATA_PATHS[1]}"
        )

    # 2) Ensure required columns are present
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = None

    # 3) Soft types
    for c in ("minutes", "n_steps", "n_ingredients"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "submitted" in df.columns and df["submitted"].notna().any():
        df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")

    return df

def validate_schema(df: pd.DataFrame) -> dict:
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
    return ROOT / "data" / "processed" / "recipes_clean.parquet"
