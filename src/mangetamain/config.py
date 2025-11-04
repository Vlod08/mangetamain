from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

RECIPES_CSV = "RAW_recipes.csv"
INTERACTIONS_CSV = "RAW_interactions.csv"

RECIPES_PARQUET = "recipes.parquet"
INTERACTIONS_PARQUET = "interactions.parquet"

DB_NAME = "mangetamain_db"

COUNTRIES_FILE = "countries.json"
