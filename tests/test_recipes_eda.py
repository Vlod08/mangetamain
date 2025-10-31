# tests/test_recipes_eda.py
from __future__ import annotations
import importlib
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# --- Ensure src/ is in sys.path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path and SRC.exists():
    sys.path.insert(0, str(SRC))

# scikit-learn is required by TfidfVectorizer used in RecipesEDAService
pytest.importorskip("sklearn", reason="scikit-learn is required for TF-IDF tests")

def _import_recipes_service():
    """Import the correct class from recipes_eda (not recipes_service)."""
    try:
        mod = importlib.import_module("mangetamain.core.recipes_eda")
    except ModuleNotFoundError:
        # fallback if your package is not registered (e.g., relative imports)
        mod = importlib.import_module("core.recipes_eda")
    return mod.RecipesEDAService

@pytest.fixture
def RecipesEDAService():
    return _import_recipes_service()

@pytest.fixture
def recipes_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id":         [1, 2, 2, 3],
            "name":       ["Apple Pie", "Tomato Pasta", "Tomato Pasta", "Avocado Toast"],
            "minutes":    [45, 20, 20, 10],
            "n_steps":    [8, 5, 5, 3],
            "submitted":  pd.to_datetime(["2020-01-01", "2020-02-01", "2020-02-01", "2021-06-15"]),
            "ingredients":[
                ["apple", "flour", "sugar"],
                ["tomato", "pasta", "basil"],
                ["tomato", "pasta", "garlic"],
                ["avocado", "bread", "lemon"],
            ],
            "nutrition": [
                [250.0, 10.0, 30.0, 200.0, 3.0, 2.0, 40.0],
                [500.0, 14.0,  8.0, 600.0, 13.0, 4.0, 70.0],
                [480.0, 12.0,  7.0, 580.0, 12.0, 3.5, 68.0],
                [300.0, 20.0,  2.0, 350.0,  6.0, 3.0, 25.0],
            ],
            "tags": [
                ["dessert", "baked"],
                ["italian", "quick"],
                ["italian", "garlicky"],
                ["breakfast", "quick"],
            ],
            "country": ["US", "IT", "IT", "US"],
            "season":  ["winter", "all", "all", "summer"],
        }
    )

@pytest.fixture
def svc(RecipesEDAService, recipes_df):
    s = RecipesEDAService()
    s.ds.df = recipes_df  # bypass .load()
    return s

def test_duplicates(svc):
    dup = svc.duplicates()
    assert "id" in dup and dup["id"] == 1
    assert dup.get("id_name", 0) >= 1

def test_nutrition_expand(svc):
    out = svc.nutrition()
    expected_cols = {"calories", "total_fat", "sugar", "sodium", "protein", "saturated_fat", "carbohydrates"}
    assert expected_cols.issubset(out.columns)
    assert len(out) == len(svc.ds.df)

def test_minutes_hist_and_steps_hist(svc):
    m = svc.minutes_hist(bins=3)
    s = svc.steps_hist(bins=3)
    assert {"left", "right", "count"}.issubset(m.columns)
    assert {"left", "right", "count"}.issubset(s.columns)
    assert int(m["count"].sum()) == len(svc.ds.df)
    assert int(s["count"].sum()) == len(svc.ds.df)

def test_by_year(svc):
    byy = svc.by_year()
    assert {"year", "n"}.issubset(byy.columns)
    yrs = set(byy["year"].tolist())
    assert yrs == {2020, 2021}
    assert int(byy.loc[byy["year"] == 2020, "n"].iloc[0]) == 3
    assert int(byy.loc[byy["year"] == 2021, "n"].iloc[0]) == 1

def test_apply_filters(svc):
    filtered = svc.apply_filters(minutes_range=(0, 15))
    assert filtered["minutes"].max() <= 15
    filtered2 = svc.apply_filters(steps_range=(0, 4))
    assert filtered2["n_steps"].max() <= 4

def test_top_ingredients(svc):
    top = svc.top_ingredients(k=5)
    assert {"ingredient", "count"}.issubset(top.columns)
    assert (top["ingredient"] == "tomato").any() or (top["ingredient"] == "pasta").any()

def test_get_unique_tags_column(svc):
    uniq = svc.get_unique("tags")
    # ✅ accepts tuple or list
    assert isinstance(uniq, (tuple, list))
    # ensure join is safe if elements are not strings
    joined = " ".join(map(str, uniq))
    for tag in ["dessert", "italian", "breakfast", "quick"]:
        assert tag in joined

def test_signatures_by_country(RecipesEDAService, recipes_df):
    df = recipes_df[["country", "ingredients"]].copy()
    sig_tfidf, sig_tf = RecipesEDAService.get_signatures_countries(df, top_n=3)
    assert set(sig_tfidf.keys()) == set(df["country"].unique())
    assert set(sig_tf.keys()) == set(df["country"].unique())
    assert all(len(v) <= 3 for v in sig_tfidf.values())
    assert all(len(v) <= 3 for v in sig_tf.values())

def test_signatures_by_season(RecipesEDAService, recipes_df):
    df = recipes_df[["season", "ingredients"]].copy()
    sig_tfidf, sig_tf = RecipesEDAService.get_signatures_seasons(df, top_n=2)
    assert set(sig_tfidf.keys()) == set(df["season"].unique())
    assert set(sig_tf.keys()) == set(df["season"].unique())
    assert all(len(v) <= 2 for v in sig_tfidf.values())
    assert all(len(v) <= 2 for v in sig_tf.values())

def test_count_recipes_seasons(RecipesEDAService, recipes_df):
    out = RecipesEDAService.count_recipes_seasons(recipes_df[["season"]].copy())
    assert {"season", "number of recipes"}.issubset(out.columns)
    got = dict(zip(out["season"], out["number of recipes"]))
    assert got.get("winter", 0) == 1
    assert got.get("summer", 0) == 1
    assert got.get("all", 0) == 2
