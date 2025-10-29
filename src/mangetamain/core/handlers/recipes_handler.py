# core/handlers/recipes_handler.py
from __future__ import annotations
from dataclasses import dataclass
import logging
import pandas as pd
from typing import List, Dict
from abc import ABC, abstractmethod
from functools import wraps
import streamlit as st

from core.utils.string_utils import fuzzy_fetch

def enforce_check(method):
    """Decorator to ensure check_ref() is called after build_ref()."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if hasattr(self, "check_ref"):
            self.check_ref()
        return result
    return wrapper

@dataclass
class RecipesHandler(ABC):
    """Abstract base class for handling recipe-related data processing.
    
    Attributes:
        ref_dataset (pd.DataFrame): Reference dataset for matching.
        ref_names (Dict[str, List[List[str]]]): Reference names for matching.
            Each key is a column name, and the value is a list of lists of reference names.
            The order of lists corresponds to different semantically related groups.
            The order is important for matching and fetching.
    """

    ref_dataset: pd.DataFrame = None
    ref_names: Dict[str, List[List[str]]] = None
    ref_path: str = None

    def __post_init__(self):
        """Initialize the RecipesHandler."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs):
        """Automatically wraps subclass build_ref with enforce_check."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "build_ref"):
            cls.build_ref = enforce_check(cls.build_ref)

    @st.cache_data(show_spinner=False)
    @abstractmethod
    def build_ref(self, path: str = None):
        pass

    def check_ref(self):
        """Check the integrity of the reference dataset and names."""
        self.logger.info("Checking reference dataset and names...")
        if self.ref_dataset is not None:
            assert isinstance(self.ref_dataset, pd.DataFrame), "Ref Dataset should be a pandas DataFrame"
            assert len(self.ref_dataset) > 0, "Ref Dataset is empty"

        assert self.ref_names is not None, "Ref Names is not properly set"
        assert len(self.ref_names) > 0, "Ref Names is empty"
        assert isinstance(self.ref_names, dict) \
            and isinstance(list(self.ref_names.keys())[0], str) \
            and isinstance(list(self.ref_names.values())[0], list) \
            and isinstance(list(self.ref_names.values())[0][0], list) \
            and isinstance(list(self.ref_names.values())[0][0][0], str), \
            "Ref Names should be a Dict[str, List[List[str]]]"

        self.logger.info("Reference dataset and names are properly set.")

    @abstractmethod
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer missing values in the DataFrame based on the reference dataset.
        Args:
            df (pd.DataFrame): The DataFrame to process.
        Returns:
            pd.DataFrame: The updated DataFrame with inferred values.
        """
        pass

    def fetch(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fetch values from the reference dataset based on the specified columns.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            columns (List[str]): List of column names to search for matching.

        Raises:
            ValueError: If a specified column is not found in the DataFrame.

        Returns:
            pd.DataFrame: The updated DataFrame with fetched values.
        """
        # Get the reference columns from the reference names
        ref_cols = list(self.ref_names.keys())

        # Create a copy of the DataFrame to avoid modifying the original during mapping
        df_copy = df.copy()

        # Iterate over each specified column
        for col_idx, col in enumerate(columns, start=1):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            
            # If the column is a string, split it into a list of tokens
            if len(df) > 0 and isinstance(df_copy[col][0], str):
                df_copy[col] = df_copy[col].str.split(expand=False)
            
            # Iterate over each reference column
            for ref_idx, ref_col in enumerate(ref_cols, start=1):
                self.logger.info(f"[{col_idx}/{len(columns)}] ({ref_idx}/{len(ref_cols)})" \
                                 + f" Processing column: {col} for {ref_col} matching...")

                # Only update rows where the reference name is empty
                if ref_col in df.columns:
                    indexes_to_update = df[df[ref_col] == ''].index
                else:
                    indexes_to_update = df.index
                df.loc[indexes_to_update, ref_col] = df_copy.loc[indexes_to_update, col].map(
                    lambda x: fuzzy_fetch(x, self.ref_names[ref_col])
                )

        # Infer missing values after fetching
        df = self.infer(df)

        return df