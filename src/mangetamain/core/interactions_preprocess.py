# src/scripts/preprocess_reviews.py
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd

RAW_REL = Path("data/raw/RAW_interactions.csv")

def find_root(start: Path | None = None) -> Path:
    here = ((start or Path(__file__)).resolve()).parent
    cands = [here, *here.parents]

    # 1) parent qui contient le csv attendu
    for c in cands:
        if (c / RAW_REL).is_file():
            return c
    # 2) un parent qui ressemble à une racine de projet
    for c in cands:
        if (c / "data").is_dir() and ((c / "src").is_dir() or (c / "pyproject.toml").is_file() or (c / ".git").exists()):
            return c
    # 3) dernier recours : le plus haut qui a data/
    for c in reversed(cands):
        if (c / "data").is_dir():
            return c
    raise FileNotFoundError("Impossible de localiser la racine (pas de dossier 'data').")

ROOT = find_root()
RAW = ROOT / RAW_REL
OUT = ROOT / "data/processed/reviews_clean.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"[preprocess-reviews] ROOT = {ROOT}")
print(f"📥 Chargement… {RAW}")
assert RAW.exists(), f"Introuvable : {RAW}"

# Lecture robuste
try:
    df = pd.read_csv(RAW, engine="pyarrow")
except Exception:
    df = pd.read_csv(RAW)

# Normalisation douce
df.columns = [c.strip().lower() for c in df.columns]
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
for c in ("user_id","recipe_id","rating"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Texte -> features basiques
if "review" in df.columns:
    s = df["review"].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["review"] = s
    df["review_len"]     = s.str.len()
    df["review_words"]   = s.str.split().map(len)
    df["exclamations"]   = s.str.count("!")
    df["question_marks"] = s.str.count(r"\?")
    df["has_caps"]       = s.str.contains(r"[A-Z]{3,}", regex=True)

# Tri/nettoyage léger
key_cols = [c for c in ["user_id","recipe_id","date","review"] if c in df.columns]
if key_cols:
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)

# Sauvegarde
print(f"💾 Sauvegarde propre → {OUT}")
df.to_parquet(OUT, index=False)
print("✅ Terminé.")

# Option : publier en DB si DATABASE_URL est défini
DB_URL = os.getenv("DATABASE_URL")
if DB_URL:
    from sqlalchemy import create_engine
    eng = create_engine(DB_URL)
    df.to_sql("reviews_clean", eng, if_exists="replace", index=False)
    print("✅ Table 'reviews_clean' écrite dans la base.")
else:
    print("ℹ️ DATABASE_URL non défini → fichiers locaux uniquement.")
