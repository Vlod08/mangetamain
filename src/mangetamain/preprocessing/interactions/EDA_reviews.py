#------------------------------------------------------------------------------
# mangetamain/preprocessing/EDA_reviews.py
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from __future__ import annotations # Enable postponed evaluation of annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import io
import json
import re
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# ReviewsPreprocessor
#------------------------------------------------------------------------------
@dataclass
class ReviewsPreprocessor:
    """
    A single-class, dependency-light preprocessing pipeline for the Food.com
    interactions dataset (or any similar reviews table).

    It encapsulates the preprocessing logic loading, normalizing schema, coercing types,
    text cleaning, simple text features, basic time features, and export of a
    minimal clean table.

    Parameters
    ----------
    path : str | Path
        Default path to the interactions file (CSV or Parquet). Used by
        `load_dataset()` if no DataFrame is provided.
    text_column : str
        Name of the review text column.
    date_column : str
        Name of the date column.
    user_col : str
        Name of the user id column.
    item_col : str
        Name of the item/recipe id column.
    rating_col : str
        Name of the rating/score column.
    stopwords : Iterable[str]
        Basic stopwords for cheap token-based filtering in `compute_text_features`.
    out_dir : str | Path
        Output directory for artifacts and exports.
    """
    # Default parameters
    path: Path = Path("../../data/RAW_interactions.csv")
    text_column: str = "review"
    date_column: str = "date"
    user_col: str = "user_id"
    item_col: str = "recipe_id"
    rating_col: str = "rating"
    # a small set of common English stopwords because the stopwords of nltk are too restrictive
    stopwords: Iterable[str] = field(default_factory=lambda: {  
        "the","a","an","and","or","is","it","to","for","of","on","in","with",
        "this","that","these","those","very","really","so","just","i","we","you",
        "he","she","they","was","were","be","been","are","am","thanks","thank",
        "recipe"
    })
    out_dir: Path = Path("../../data/")
#------------------------------------------------------------------------------
# ReviewsPreprocessor Methods (External)
#------------------------------------------------------------------------------
    
    def load_dataset(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load CSV/Parquet into a DataFrame.

        If `path` is None, uses the instance's `path`.
        """
        if path is None:
            path = self.path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        try:
            return pd.read_csv(p, engine="pyarrow")  # more quick and robust
        except Exception:
            return pd.read_csv(p)
        
    def save_csv(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        df.to_csv(path, index=False)

    def save_text(self, text: str, path: Path) -> None:
        """Save text to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        path.write_text(text)

    def save_json(self, obj: dict, path: Path) -> None:
        """Save JSON object to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        path.write_text(json.dumps(obj, indent=2))

#------------------------------------------------------------------------------
# ReviewsPreprocessor Methods (Internal)
#------------------------------------------------------------------------------

    def _normalize_column_names(self, df: pd.DataFrame) -> None:
        """Normalize column names to lowercase stripped strings."""
        df.columns = [str(c).strip().lower() for c in df.columns]

    def _coerce_types(self, df: pd.DataFrame) -> None:
        """Coerce date and numeric columns (user_id, recipe_id, rating) to appropriate types."""
        if self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        for col in [self.user_col, self.item_col, self.rating_col]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    def _clean_text(self, df: pd.DataFrame) -> None:
        """Basic text cleaning for reviews: normalize whitespace and strip."""
        df[self.text_column] = (
            df[self.text_column]
              .astype("string")
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
        )

