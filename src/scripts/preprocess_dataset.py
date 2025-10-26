# scripts/preprocess_dataset.py
from __future__ import annotations
from pathlib import Path
import os, ast
import numpy as np
import pandas as pd

RAW_REL = Path("data/raw/RAW_recipes.csv")

def find_root(start: Path | None = None) -> Path:
    """Trouve la racine du repo en remontant et en privilégiant un parent
    qui contient data/raw/RAW_recipes.csv. Ignore les faux 'data/' locaux."""
    here = ((start or Path(__file__)).resolve()).parent
    candidates = [here, *here.parents]

    # 1) parent qui contient le fichier attendu
    for cand in candidates:
        if (cand / RAW_REL).is_file():
            return cand

    # 2) parent qui ressemble à une racine (data + src ou pyproject/.git)
    for cand in candidates:
        if (cand / "data").is_dir() and (
            (cand / "src").is_dir() or
            (cand / "pyproject.toml").is_file() or
            (cand / ".git").exists()
        ):
            return cand

    # 3) dernier recours: le plus haut qui a un 'data/'
    for cand in reversed(candidates):
        if (cand / "data").is_dir():
            return cand

    raise FileNotFoundError("Impossible de localiser la racine du projet (pas de dossier 'data').")

ROOT = find_root()
print(f"[preprocess] ROOT = {ROOT}")

RAW = ROOT / RAW_REL                     # brut
OUT = ROOT / "data/processed/recipes_clean.parquet"  # propre
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"📥 Chargement… {RAW}")
assert RAW.exists(), f"Introuvable : {RAW}"
df = pd.read_csv(RAW)

# ---- Types de base
for c in ["minutes", "n_steps", "n_ingredients"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")

# ---- Dates
df["submitted"] = pd.to_datetime(df.get("submitted"), errors="coerce")

# ---- Tags / ingredients -> listes propres
def to_list(x):
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return [str(t).strip().lower() for t in v]
    except Exception:
        pass
    return []
df["tags"] = df.get("tags", "").apply(to_list)
df["ingredients"] = df.get("ingredients", "").apply(to_list)

# ---- Description textuelle
df["description"] = df.get("description", "").fillna("").astype(str)

# ---- Nutrition (si présente) -> colonnes séparées
if "nutrition" in df.columns:
    def split_nut(x):
        try:
            v = ast.literal_eval(x); assert isinstance(v, list)
            v = (v + [np.nan]*7)[:7]
            return v
        except Exception:
            return [np.nan]*7
    cols = ["calories","total_fat","sugar","sodium","protein","saturated_fat","carbohydrates"]
    nut = pd.DataFrame(df["nutrition"].apply(split_nut).tolist(), columns=cols, index=df.index)
    df = pd.concat([df.drop(columns=["nutrition"]), nut], axis=1)

# ---- Nettoyage basique
df = df.drop_duplicates(subset=["name","description"])
df = df[df["minutes"].notna()].reset_index(drop=True)

# ---- Sauvegarde Parquet
print(f"💾 Sauvegarde propre → {OUT}")
df.to_parquet(OUT, index=False)
print("✅ Terminé.")

# ---- (optionnel) publier en DB si DATABASE_URL est défini
DB_URL = os.getenv("DATABASE_URL")
if DB_URL:
    from sqlalchemy import create_engine
    eng = create_engine(DB_URL)
    df.to_sql("recipes_clean", eng, if_exists="replace", index=False)
    print("✅ Table 'recipes_clean' écrite dans la base.")
else:
    print("ℹ️ DATABASE_URL non défini → on reste sur fichiers.")
