# scripts/preprocess_dataset.py
import pandas as pd, numpy as np, ast
from pathlib import Path

RAW = Path("./RAW_recipes.csv")                     # <-- brut
OUT = Path("./data/recipes_clean.parquet")     # <-- propre
OUT.parent.mkdir(parents=True, exist_ok=True)

print("📥 Chargement…")
df = pd.read_csv(RAW)

# Types de base
for c in ["minutes", "n_steps", "n_ingredients"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")

# Dates
df["submitted"] = pd.to_datetime(df.get("submitted"), errors="coerce")

# Tags / ingredients -> listes propres
def to_list(x):
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list): return [str(t).strip().lower() for t in v]
    except Exception: pass
    return []
df["tags"] = df.get("tags", "").apply(to_list)
df["ingredients"] = df.get("ingredients", "").apply(to_list)

# Description textuelle
df["description"] = df.get("description", "").fillna("").astype(str)

# Nutrition (si présente) -> colonnes
if "nutrition" in df.columns:
    def split_nut(x):
        try:
            v = ast.literal_eval(x); assert isinstance(v, list)
            v = (v + [np.nan]*7)[:7]  # pad/tronc
            return v
        except Exception: return [np.nan]*7
    cols = ["calories","total_fat","sugar","sodium","protein","saturated_fat","carbohydrates"]
    nut = pd.DataFrame(df["nutrition"].apply(split_nut).tolist(), columns=cols)
    df = pd.concat([df.drop(columns=["nutrition"]), nut], axis=1)

# Drop doublons grossiers + lignes sans minutes
df = df.drop_duplicates(subset=["name","description"])
df = df[df["minutes"].notna()].reset_index(drop=True)

print(f"💾 Sauvegarde propre: {OUT}")
df.to_parquet(OUT, index=False)
print("✅ Terminé.")
