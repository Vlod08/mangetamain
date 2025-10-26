# src/core/recipes_preprocessor.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import ast, os
import numpy as np
import pandas as pd

@dataclass
class RecipesPreprocessor:
    anchor: Path  # Path(__file__) du caller (page)
    raw_rel: Path = Path("data/raw/RAW_recipes.csv")
    out_rel: Path = Path("data/processed/recipes_clean.parquet")

    def _root(self) -> Path:
        p = self.anchor.resolve()
        for cand in [p, *p.parents]:
            if (cand / "data").is_dir():
                return cand
        return p

    @property
    def RAW(self) -> Path: return self._root() / self.raw_rel
    @property
    def OUT(self) -> Path: return self._root() / self.out_rel

    def run(self) -> Path:
        RAW, OUT = self.RAW, self.OUT
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
                ...
            return []
        if "tags" in df.columns:        df["tags"] = df["tags"].apply(to_list)
        if "ingredients" in df.columns: df["ingredients"] = df["ingredients"].apply(to_list)

        # ---- Description
        df["description"] = df.get("description", "").fillna("").astype(str)

        # ---- Nutrition -> colonnes
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

        # ---- Nettoyage
        df = df.drop_duplicates(subset=["name","description"])
        df = df[df["minutes"].notna()].reset_index(drop=True)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)

        # (optionnel) publier DB si DATABASE_URL
        url = os.getenv("DATABASE_URL")
        if url:
            from sqlalchemy import create_engine
            pd.io.sql.to_sql(df, name="recipes_clean", con=create_engine(url), if_exists="replace", index=False)

        return OUT
