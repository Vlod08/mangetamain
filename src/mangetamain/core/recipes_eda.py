# core/recipes_service.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import logging
from abc import ABC, abstractmethod

from mangetamain.core.dataset import COUNTRIES_FILE_PATH, DatasetLoader, RecipesDataset
from mangetamain.core.utils.string_utils import extract_classes
from mangetamain.core.handlers.country_handler import CountryHandler
from mangetamain.core.handlers.seasonnality_handler import SeasonalityHandler


@dataclass
class EDAService(ABC):
    """Base class for EDA services."""

    ds: DatasetLoader = field(default_factory=DatasetLoader)
    label: str = ""

    def __post_init__(self):
        """Initialize the EDA service."""
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.load()

    def load(self, df: pd.DataFrame = None, preprocess: bool = True) -> pd.DataFrame:
        """Load the dataset.
        Args:
            df (pd.DataFrame, optional): DataFrame to load. If None, load from default path.
            preprocess (bool): Whether to preprocess the dataset after loading.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        # If a DataFrame is provided, use it directly
        if df is not None and not df.empty:
            # Update the dataset in the service
            if preprocess:
                df = self.ds.preprocess(df)
                st.session_state["issues"][self.label] = self.ds.issues
                return df
            else:
                # Load issues from session state if available
                self.ds.df = df
                if self.label in st.session_state.get("issues", {}):
                    self.ds.issues = st.session_state["issues"][self.label]
                return df
        # Otherwise, load from the default path
        df = self.ds.load(preprocess=preprocess)
        if preprocess:
            st.session_state["issues"][self.label] = self.ds.issues
        return df

    # ---------- Data Quality ----------
    def schema(self) -> pd.DataFrame:
        """Get the schema of the dataset."""
        return self.ds.schema

    def na_counts(self) -> pd.Series:
        """Get the NA counts of each column."""
        # Check if issues are stored in the dataset or in session state
        if self.ds.issues and "nan" in self.ds.issues:
            nan_counts = self.ds.issues["nan"]
        # Check if issues are stored in session state
        else:
            if self.label not in st.session_state.get("issues", {}):
                return pd.Series(dtype=int)
            nan_counts = st.session_state["issues"][self.label].get("nan", {})

        nan_series = pd.Series(nan_counts)
        nan_series = nan_series.sort_values(ascending=False)
        nan_series = nan_series.rename("Na count")
        return nan_series

    def na_rate(self) -> pd.Series:
        """Get the NA rate of each column."""
        # Check if issues are stored in the dataset or in session state
        if self.ds.issues and "nan" in self.ds.issues:
            nan_counts = self.ds.issues["nan"]
        # Check if issues are stored in session state
        else:
            if self.label not in st.session_state.get("issues", {}):
                return pd.Series(dtype=float)
            nan_counts = st.session_state["issues"][self.label].get("nan", {})

        nan_rate = pd.Series(
            {col: count / len(self.ds.df) for col, count in nan_counts.items()}
        )

        nan_rate_percent = nan_rate * 100
        nan_rate_percent = nan_rate_percent.sort_values(ascending=False)
        nan_rate_percent = nan_rate_percent.rename("Na rate (%)")
        return nan_rate_percent

    @abstractmethod
    def duplicates(self) -> dict:
        """Get the duplicate rows in the dataset."""
        raise NotImplementedError("Subclasses must implement the duplicates method.")

    def numeric_desc(self) -> pd.DataFrame:
        return self.ds.df.select_dtypes("number").describe().T

    def cardinalities(self) -> pd.Series:
        """Get the cardinality of each categorical column."""
        # convert only object-like columns; numeric/datetime are fine as-is
        obj_cols = self.ds.df.select_dtypes(include=["object"]).columns
        df_copy = self.ds.df[obj_cols].copy()
        for c in obj_cols:
            df_copy[c] = df_copy[c].map(DatasetLoader.to_hashable)

        cards = df_copy.nunique(dropna=True).sort_values(ascending=False)
        cards = cards.rename("cardinality")
        return cards

    def export_clean_min(self, path: Path = None) -> pd.DataFrame:
        """Export a minimal clean DataFrame."""
        self.ds.export(path=path)


@dataclass
class RecipesEDAService(EDAService):

    ds: RecipesDataset = field(default_factory=RecipesDataset)
    label: str = "recipes"
    country_handler: CountryHandler = field(
        default_factory=lambda: CountryHandler(ref_path=COUNTRIES_FILE_PATH)
    )
    season_handler = SeasonalityHandler()

    def duplicates(self) -> dict:
        """
        Check for duplicates in the recipes dataset.
        Works even when some columns are lists/ndarrays/dicts.
        Returns:
            dict: A dictionary with counts of duplicates by different criteria.
                  Keys can include 'id', 'id_name', and 'full' for different levels of duplication.
                  Returns an empty dict if no duplicates are found.
        """

        return_duplicates = {}
        id_duplicates = int(self.ds.df["id"].duplicated().sum())
        if id_duplicates > 0:
            return_duplicates["id"] = id_duplicates
            id_name_duplicates = int(sum(self.ds.df.duplicated(["id", "name"])))
            if id_name_duplicates > 0:
                return_duplicates["id_name"] = id_name_duplicates
                # Hashable view for duplicate detection
                df_hashable = self.ds.df.map(DatasetLoader.to_hashable)
                return_duplicates["full"] = int(sum(df_hashable.duplicated()))
            return return_duplicates
        return return_duplicates

    def get_unique(self, column: str) -> tuple[str]:
        """Compute unique values across all recipes.
        Args:
            column (str): The column name to extract unique values from.
        Returns:
            List[str]: A sorted list of unique values.
        """
        if column not in self.ds.df.columns:
            self.logger.warning(f"Column '{column}' not found in dataset.")
            return ()
        return extract_classes(self.ds.df[column])

    # ---------- Data Fetching ----------
    def fetch_country(self, df) -> pd.DataFrame:
        """Fetch recipes with country information.
        Returns:
            pd.DataFrame: DataFrame containing recipes with country information.
        """
        self.country_handler.build_ref()
        columns_to_fetch_in = ["tags"]  # + ["name", "description"]
        df_country = self.country_handler.fetch(df, columns_to_fetch_in)
        if "country" not in df_country.columns:
            self.logger.warning("Column 'country' not found in dataset.")
            return pd.DataFrame()
        return df_country[df_country["country"] != ""]

    def fetch_period(self, df) -> pd.DataFrame:
        """Fetch recipes with season information.
        Returns:
            pd.DataFrame: DataFrame containing recipes with season information.
        """
        df_period = self.season_handler.get_periods_all(df)
        if "season" not in df_period.columns:
            self.logger.warning("Column 'season' not found in dataset.")
            return pd.DataFrame()
        if "event" not in df_period.columns:
            self.logger.warning("Column 'event' not found in dataset.")
            return pd.DataFrame()
        return df_period[(df_period["season"] != "") | (df_period["event"] != "")]

    # ---------- Explorer helpers ----------
    def nutrition(self) -> pd.DataFrame:
        """Expand the nutrition column into separate columns."""
        if "nutrition" in self.ds.df.columns:
            df = self.ds.df.copy()
            cols = [
                "calories",
                "total_fat",
                "sugar",
                "sodium",
                "protein",
                "saturated_fat",
                "carbohydrates",
            ]
            nut = pd.DataFrame(df["nutrition"].tolist(), columns=cols, index=df.index)
            df = pd.concat([df.drop(columns=["nutrition"]), nut], axis=1)
            return df
        return pd.DataFrame()

    def minutes_hist(self, bins: int = 40) -> pd.DataFrame:
        """Get histogram of cooking minutes in recipes."""
        if self.ds.df["minutes"].empty:
            return pd.DataFrame()
        c, e = np.histogram(self.ds.df["minutes"], bins=bins)
        return pd.DataFrame({"left": e[:-1], "right": e[1:], "count": c})

    def steps_hist(self, bins: int = 25) -> pd.DataFrame:
        """Get histogram of number of steps in recipes."""
        if self.ds.df["n_steps"].empty:
            return pd.DataFrame()
        c, e = np.histogram(self.ds.df["n_steps"], bins=bins)
        return pd.DataFrame({"left": e[:-1], "right": e[1:], "count": c})

    def by_year(self) -> pd.DataFrame:
        """Get the number of recipes submitted by year."""
        df = self.ds.df.copy()
        if "submitted" not in df.columns:
            return pd.DataFrame()
        df["year"] = df["submitted"].dt.year
        return (
            df.dropna(subset=["year"])
            .groupby("year")
            .size()
            .reset_index(name="n")
            .sort_values("year")
        )

    def apply_filters(
        self,
        minutes: tuple[int, int] | None = None,
        steps: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """
        Apply various filters to the recipes dataset.
        """
        # Start with the full dataset
        # Make a copy to avoid modifying the original DataFrame
        df = self.ds.df.copy()
        cols = df.columns.tolist()

        if minutes and "minutes" in cols:
            lo, hi = minutes
            df = df[(df["minutes"] >= lo) & (df["minutes"] <= hi)]

        if steps and "n_steps" in cols:
            lo, hi = steps
            df = df[(df["n_steps"] >= lo) & (df["n_steps"] <= hi)]

        return df[cols]

    def top_ingredients(self, k: int = 30) -> pd.DataFrame:
        """Get the top-k ingredients across all recipes."""
        if "ingredients" not in self.ds.df.columns:
            self.logger.warning("Column 'ingredients' not found in dataset.")
            return pd.DataFrame(columns=["ingredient", "count"])
        cnt = Counter()
        for row in self.ds.df["ingredients"].dropna():
            if isinstance(row, list):
                cnt.update([str(x).lower().strip() for x in row if x])
        return pd.DataFrame(cnt.most_common(k), columns=["ingredient", "count"])

    @staticmethod
    def get_signatures_countries(df: pd.DataFrame, top_n: int = 10) -> dict:
        """Compute signature ingredients per country using TF-IDF.
        Args:
            df (pd.DataFrame): DataFrame containing 'country' and 'ingredients' columns.
            top_n (int): Number of top ingredients to include in the signature.
        Returns:
            dict: A dictionary mapping each country to its signature ingredients and their TF-IDF scores.
        """
        if "country" not in df.columns or "ingredients" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'country' and 'ingredients' columns."
            )

        # Aggregate all ingredient lists per country into a single list per country
        country_docs_lists = df.groupby("country")["ingredients"].sum()

        base_params = {
            "preprocessor": lambda x: x,
            "tokenizer": lambda x: x,
            "lowercase": False,
            "max_df": 0.5,
            "max_features": 14000,
            "token_pattern": None,
        }

        # 3) TF-IDF (fit) to build the vocabulary and compute TF-IDF scores
        tfidf_vec = TfidfVectorizer(**base_params)
        tfidf_matrix = tfidf_vec.fit_transform(country_docs_lists)
        features = tfidf_vec.get_feature_names_out()

        # 4) L1-normalized TF (transform) using the same vocabulary
        tf_vec = TfidfVectorizer(
            **base_params, use_idf=False, norm="l1", vocabulary=tfidf_vec.vocabulary_
        )
        tf_matrix = tf_vec.fit_transform(country_docs_lists)

        # 5) Build convenient DataFrames for TF-IDF and TF matrices
        countries = country_docs_lists.index
        df_tfidf = pd.DataFrame(
            tfidf_matrix.toarray(), index=countries, columns=features
        )
        df_tf = pd.DataFrame(tf_matrix.toarray(), index=countries, columns=features)

        # 6) Build output: top_n terms by TF-IDF and include TF values
        signatures_tfidf, signatures_tf = {}, {}
        for country in df_tfidf.index:
            top_terms = df_tfidf.loc[country].nlargest(top_n).index
            signatures_tfidf[country] = {
                term: float(df_tfidf.at[country, term]) for term in top_terms
            }
            signatures_tf[country] = {
                term: float(df_tf.at[country, term]) for term in top_terms
            }

        return signatures_tfidf, signatures_tf

    @staticmethod
    def get_signatures_seasons(df: pd.DataFrame, top_n: int = 10) -> dict:
        """Compute signature ingredients per season using TF-IDF.
        Args:
            df (pd.DataFrame): DataFrame containing 'season' and 'ingredients' columns.
            top_n (int): Number of top ingredients to include in the signature.
        Returns:
            dict: A dictionary mapping each season to its signature ingredients and their TF-IDF scores.
        """
        if "season" not in df.columns or "ingredients" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'season' and 'ingredients' columns."
            )

        # Aggregate all ingredient lists per season into a single list per season
        season_docs_lists = df.groupby("season")["ingredients"].sum()

        base_params = {
            "preprocessor": lambda x: x,
            "tokenizer": lambda x: x,
            "lowercase": False,
            "max_df": 0.5,
            "max_features": 14000,
            "token_pattern": None,
        }

        # 3) TF-IDF (fit) to build the vocabulary and compute TF-IDF scores
        tfidf_vec = TfidfVectorizer(**base_params)
        tfidf_matrix = tfidf_vec.fit_transform(season_docs_lists)
        features = tfidf_vec.get_feature_names_out()

        # 4) L1-normalized TF (transform) using the same vocabulary
        tf_vec = TfidfVectorizer(
            **base_params, use_idf=False, norm="l1", vocabulary=tfidf_vec.vocabulary_
        )
        tf_matrix = tf_vec.fit_transform(season_docs_lists)

        # 5) Build convenient DataFrames for TF-IDF and TF matrices
        seasons = season_docs_lists.index
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=seasons, columns=features)
        df_tf = pd.DataFrame(tf_matrix.toarray(), index=seasons, columns=features)

        # 6) Build output: top_n terms by TF-IDF and include TF values
        signatures_tfidf, signatures_tf = {}, {}
        for season in df_tfidf.index:
            top_terms = df_tfidf.loc[season].nlargest(top_n).index
            signatures_tfidf[season] = {
                term: float(df_tfidf.at[season, term]) for term in top_terms
            }
            signatures_tf[season] = {
                term: float(df_tf.at[season, term]) for term in top_terms
            }

        return signatures_tfidf, signatures_tf

    @staticmethod
    def count_recipes_seasons(df_period: pd.DataFrame):
        seasons_counts = (
            df_period[df_period["season"] != ""]["season"].value_counts().reset_index()
        )
        seasons_counts.columns = ["season", "number of recipes"]
        return seasons_counts
