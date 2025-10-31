# core/dataset.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import logging
from abc import ABC
import re
import numpy as np
from threading import Thread

from mangetamain.config import ROOT_DIR
from mangetamain.core.utils.string_utils import (
    extract_list_strings, extract_list_floats, 
    looks_like_datetime, is_list_string, is_list_floats_string
)
from mangetamain.core.app_logging import get_logger

DATA_DIR: Path = ROOT_DIR / "data"
MANGETAMAIN_DB_PATH: Path = DATA_DIR / "mangetamain.db"
RECIPES_RAW_DATA_PATH: Path = DATA_DIR / "RAW_recipes.csv"
RECIPES_PARQUET_DATA_PATH: Path = DATA_DIR / "processed" / "recipes.parquet"
INTERACTIONS_RAW_DATA_PATH: Path = DATA_DIR / "RAW_interactions.csv"
INTERACTIONS_PARQUET_DATA_PATH: Path = DATA_DIR / "processed" / "interactions.parquet"
COUNTRIES_FILE_PATH: Path = DATA_DIR / "countries.json"

class DatasetLoaderThread(Thread):
    def __init__(self, dataset_loader: DatasetLoader, label: str = ""):# , target = None):
        super().__init__()
        self.dataset_loader = dataset_loader
        # self.target = target
        self.label = label
        self.return_value = pd.DataFrame()
        self.issues = {}

    def run(self):
        self.return_value = self.dataset_loader.load(preprocess=True)
        self.issues = self.dataset_loader.issues

@dataclass
class DatasetLoader(ABC):

    table: str = ""
    _df: pd.DataFrame | None = None
    _schema: pd.DataFrame | None = None
    issues: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = get_logger(self.__class__.__name__)

    # ---- DataFrame Property ----
    @property
    def df(self) -> pd.DataFrame:
        """Return the loaded dataset."""
        if self._df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame | None) -> None:
        """Set the DataFrame and automatically update schema and preprocess."""
        if value is not None:
            if not isinstance(value, pd.DataFrame):
                raise ValueError("df must be a pandas DataFrame.")
            self._df = value
            # Automatically compute schema
            self._schema = DatasetLoader.compute_schema(self._df)
            # Automatically preprocess
            # self._df = self.preprocess(self._df)
        else:
            # Clear both df and schema
            self._df = None
            self._schema = None

    # ---- Schema Property ----
    @property
    def schema(self) -> pd.DataFrame:
        """Return schema information about the dataset."""
        if self._schema is None:
            raise ValueError("Schema not available. Dataset not loaded.")
        return self._schema
    
    @staticmethod
    def compute_schema(df: pd.DataFrame) -> pd.DataFrame:
        """Compute and return schema information about the dataset."""
        if df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        schema_df = pd.DataFrame({
            "col": df.columns,
            "dtype": df.dtypes.astype(str)
        })
        return schema_df

    # (No setter for schema — it's derived automatically from df)

    # ---- Loading Methods ----
    @staticmethod
    def load_files(data_name: str, _logger: logging.Logger) -> pd.DataFrame:
        """Load dataset from CSV or Parquet files."""
        _logger.info(f"Attempting to load {data_name} dataset...")
        if data_name == "recipes":
            parquet_path = RECIPES_PARQUET_DATA_PATH
            raw_path = RECIPES_RAW_DATA_PATH
        elif data_name == "interactions":
            parquet_path = INTERACTIONS_PARQUET_DATA_PATH
            raw_path = INTERACTIONS_RAW_DATA_PATH
        else:
            raise ValueError(f"Unknown data_name: {data_name}")

        if parquet_path.exists():
            _logger.info(f"Loading {data_name} from {parquet_path}...")
            return pd.read_parquet(parquet_path)

        if raw_path.exists():
            _logger.info(f"Loading {data_name} from {raw_path}...")
            return pd.read_csv(raw_path)

        _logger.error(f"Failed to load {data_name}: {parquet_path} or {raw_path} not found")
        raise FileNotFoundError(f"Not found: {parquet_path} or {raw_path}")

    @staticmethod
    def load_db(data_name: str, _logger: logging.Logger) -> pd.DataFrame | None:
        """Load dataset from SQLite database."""
        url = f"sqlite:///{MANGETAMAIN_DB_PATH}"
        _logger.info(f"Attempting to load {data_name} dataset from DB at {url}...")
        if not MANGETAMAIN_DB_PATH.exists():
            _logger.warning("Database file not found.")
            return None
        try:
            eng = create_engine(url, pool_pre_ping=True)
            return pd.read_sql(f"SELECT * FROM {data_name}", eng)
        except Exception as e:
            _logger.error(f"Failed to load {data_name} dataset from DB: {e}")
            return None

    # we tell Streamlit not to hash this argument by adding a 
    # leading underscore to the argument's name in the function signature
    @st.cache_data(show_spinner=False)
    @staticmethod
    def load_dataset(table: str, _logger: logging.Logger) -> pd.DataFrame:
        """Load the dataset into self.df.
        Args:
            table (str): The name of the table/dataset to load.
            _logger (logging.Logger): Logger for logging messages.
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if not table:
            raise NotImplementedError("Subclasses must define the table name.")
        _logger.info("Loading dataset ...")

        df = None  # reset
        try:
            df = DatasetLoader.load_db(table, _logger)
            if df is None:
                raise ValueError("DB returned no data")
        except Exception as e:
            _logger.warning(f"DB load failed ({e}), trying files...")
            df = DatasetLoader.load_files(table, _logger)
        finally:
            if df is not None:
                _logger.info(f"{table} dataset loaded successfully.")
            else:
                _logger.error(f"Failed to load {table} dataset from both DB and files.")
        # df = self.preprocess(df)
        # self.df = df  # triggers preprocessing and schema computation
        return df

    def load(self, preprocess: bool = True) -> None:
        """Load the dataset into self.df.
        Args:
            preprocess (bool): Whether to preprocess the dataset after loading.
        """
        df = DatasetLoader.load_dataset(self.table, self.logger)
        if preprocess:
            df = self.preprocess(df)
        self.df = df
        return df

    # ---- Preprocessing Methods ----
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset after loading."""
        if df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        df_copy = df.copy()
        self.logger.info(f"Preprocessing {self.__class__.__name__} dataset...")
        self.issues = {'nan': {}}

        numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
        date_cols = df_copy.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        object_cols = df_copy.select_dtypes(include=['object']).columns.tolist()

        for col in numeric_cols:
            df_copy[col] = self._preprocess_numeric_column(df_copy[col])

        for col in date_cols:
            df_copy[col] = self._preprocess_date_column(df_copy[col])

        for col in object_cols:
            # Check if the column contains list-like strings
            col_no_nans = df_copy[col].dropna()
            if col_no_nans.empty:
                continue
            if looks_like_datetime(col_no_nans.values[0]):
                df_copy[col] = self._preprocess_date_column(df_copy[col])
            elif is_list_floats_string(col_no_nans.values[0]):
                df_copy[col] = self._process_list_floats_column(df_copy[col])
            elif is_list_string(col_no_nans.values[0]):
                df_copy[col] = self._preprocess_list_strings_column(df_copy[col])
            else:
                df_copy[col] = self._preprocess_string_column(df_copy[col])

        self.logger.info("Preprocessing completed.")
        # df_copy = DatasetLoader.make_dataframe_hashable(df_copy)
        return df_copy

    def _preprocess_list_strings_column(
        self,
        column: pd.Series
    ) -> pd.Series:
        """
        Preprocess a column containing string representations of lists.

        Args:
            column (pd.Series): The column containing string representations of lists.
        Returns:
            pd.Series: The processed column as a Series of lists of strings.
        """
        self.issues['nan'][column.name] = column.isnull().sum()
        processed_column = column.fillna('[]')
        processed_column = processed_column.astype(str)
        processed_column = processed_column.str.lower().str.strip()
        processed_column = processed_column.apply(
            lambda s: re.sub(r'[^a-z0-9\s\'\"\[\]\,-]', '', s))
        processed_column = processed_column.apply(
            lambda s: s if not re.search(r"less_than|greater_than|sql", s) else None)
        processed_column = processed_column.apply(extract_list_strings)
        return processed_column

    def _process_list_floats_column(
        self,
        column: pd.Series
    ) -> pd.Series:
        """
        Preprocess a column containing string representations of lists of floats.

        Args:
            column (pd.Series): The column containing string representations of lists of floats.
        Returns:
            pd.Series: The processed column as a Series of lists of floats.
        """
        self.issues['nan'][column.name] = column.isnull().sum()
        processed_column = column.fillna('[]')
        processed_column = processed_column.astype(str)
        processed_column = processed_column.apply(extract_list_floats)
        return processed_column

    def _preprocess_string_column(
        self,
        column: pd.Series
    ) -> pd.Series:
        """
        Preprocess a string column in the recipes.
        Args:
            column (pd.Series): The column to preprocess.
        Returns:
            pd.Series: The processed string column.
        """
        self.issues['nan'][column.name] = column.isnull().sum()
        processed_column = column.fillna('')
        processed_column = processed_column.astype(str)
        processed_column = processed_column.str.lower().str.strip()
        return processed_column

    def _preprocess_numeric_column(
        self,
        column: pd.Series
    ) -> pd.Series:
        """
        Preprocess a numeric column in the recipes.
        Args:
            column (pd.Series): The column to preprocess.
        Returns:
            pd.Series: The processed numeric column.
        """
        processed_column = pd.to_numeric(column, errors='coerce')
        self.issues['nan'][column.name] = column.isnull().sum()
        return processed_column

    def _preprocess_date_column(
        self,
        column: pd.Series
    ) -> pd.Series:
        """
        Preprocess a date column in the recipes.
        Args:
            column (pd.Series): The column to preprocess.
        Returns:
            pd.Series: The processed date column.
        """
        # Convert to datetime, extract date, handle errors
        processed_column = pd.to_datetime(column, errors='coerce')
        self.issues['nan'][column.name] = column.isnull().sum()
        # processed_column = processed_column.dt.date
        processed_column = processed_column.fillna(pd.NaT)
        return processed_column

    # ---- Export Methods ----
    def export_parquet(self, path: Path) -> None:
        """Export DataFrame to Parquet."""
        if not path:
            if not self.table:
                raise ValueError("Table name not defined for default export path.")
            output_dir = ROOT_DIR / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{self.table}_clean.parquet"
        else:
            output_file = path
        self.df.to_parquet(output_file, index=False)
        self.logger.info(f"DataFrame exported to Parquet at {output_file}.")

    def export_sql(self, db_url: str) -> None:
        """Export DataFrame to SQL database."""
        if not db_url:
            if not self.table:
                raise ValueError("Table name not defined for default export path.")
            output_dir = ROOT_DIR / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            db_path = output_dir / f"{self.table}_clean.db"
            db_url = f"sqlite:///{db_path}"  # Example; replace with actual config
        else:
            db_path = db_url.replace("sqlite:///", "")
        eng = create_engine(db_url)
        self.df.to_sql(f"{self.table}_clean", eng, if_exists="replace", index=False)
        self.logger.info(f"DataFrame exported to SQL database at {db_url}.")

    def export_csv(self, path: Path = None) -> None:
        """Export DataFrame to CSV."""
        if not path:
            if not self.table:
                raise ValueError("Table name not defined for default export path.")
            output_dir = ROOT_DIR / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{self.table}_clean.csv"
        else:
            output_file = path
        self.df.to_csv(output_file, index=False)
        self.logger.info(f"DataFrame exported to CSV at {output_file}.")

    def export(self, format: str = None, path: Path = None) -> None:
        """Export DataFrame to specified format."""
        format = format.lower()
        if format is None:
            self.export_parquet(path)
            self.export_sql(str(path))
        elif format == "parquet":
            self.export_parquet(path)
        elif format == "sql":
            self.export_sql(str(path))
        elif format == "csv":
            self.export_csv(path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # ---- Utility Methods ----
    @st.cache_data(show_spinner=False)
    @staticmethod
    def make_dataframe_hashable(df: pd.DataFrame) -> pd.DataFrame:
        """Convert non-hashable objects (lists, dicts, arrays) into hashable tuples for caching."""
        hashable_df = df.copy()
        for col in df.columns:
            if DatasetLoader.is_non_hashable_column(df[col]):
                hashable_df[col] = df[col].apply(DatasetLoader.to_hashable)
        return hashable_df

    @staticmethod
    def is_non_hashable_column(series: pd.Series) -> bool:
        """Check if a column contains unhashable types."""
        return series.apply(lambda x: not isinstance(x, (int, float, str, bool, type(None), np.generic))).any()

    @staticmethod
    def to_hashable(x, _depth=0):
        """Safely convert to hashable type."""
        if _depth > 10:
            return str(x)
        if isinstance(x, np.ndarray):
            return tuple(DatasetLoader.to_hashable(v, _depth + 1) for v in x.tolist())
        if isinstance(x, (list, tuple, set)):
            return tuple(DatasetLoader.to_hashable(v, _depth + 1) for v in x)
        if isinstance(x, dict):
            return tuple((k, DatasetLoader.to_hashable(v, _depth + 1)) for k, v in sorted(x.items()))
        return x

@dataclass
class RecipesDataset(DatasetLoader):
    table: str = "recipes"

@dataclass
class InteractionsDataset(DatasetLoader):
    table: str = "interactions"
