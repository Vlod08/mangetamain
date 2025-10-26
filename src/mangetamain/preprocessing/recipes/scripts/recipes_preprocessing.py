import logging
from typing import List, Dict, Any
import pandas as pd
import re
from pathlib import Path

from utils.string_utils import extract_list_strings

class RecipesPreprocessor:
    """Preprocess recipes DataFrame."""

    issues = {}
    info = {}

    def __init__(self, recipes_df: pd.DataFrame = None):
        """RecipesPreprocessor initializer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing RecipesPreprocessor with provided DataFrame...")
        self.recipes_df = recipes_df
        self.lookup_root_dir()
        self.set_data_paths()
        self.recipes_df = pd.DataFrame()
        self.load_dataset()

    def lookup_root_dir(self) -> None:
        """
        Lookup the root directory of the repository.
        """
        current_dir = Path(__file__)
        dir_path = current_dir
        dir_name = dir_path.name
        while True:
            dir_path = dir_path.parent
            dir_name = dir_path.name
            if dir_name == "mangetamain":
                # Check if we are in a git repository
                if (dir_path / ".git").exists():
                    break
        self.REPO_ROOT = dir_path

    def set_data_paths(self) -> None:
        """
        Set the data directory and file paths.
        """
        self.DATA_DIR = self.REPO_ROOT / "data"
        self.RAW_DATA_PATH = self.DATA_DIR / "RAW_recipes.csv"
        self.COUNTRIES_FILE_PATH = self.DATA_DIR / "countries.json"

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the recipes dataset from a CSV file.
        """
        self.logger.info(f"Loading dataset from {self.RAW_DATA_PATH}...")
        self.recipes_df = pd.read_csv(self.RAW_DATA_PATH)
        self.logger.info(f"Loaded {len(self.recipes_df)} recipes.")

    def preprocess_list_strings_column(
        self,
        column: pd.Series
    ):
        """
        Preprocess a column containing string representations of lists.

        Args:
            column (pd.Series): The column containing string representations of lists.
        """
        if column not in self.issues['nan']:
            self.issues['nan'][column] = self.recipes_df[column].isnull().sum()
        self.recipes_df[column] = self.recipes_df[column].fillna('[]')
        self.recipes_df[column] = self.recipes_df[column].astype(str)
        self.recipes_df[column] = self.recipes_df[column].str.lower().str.strip()
        self.recipes_df[column] = self.recipes_df[column].apply(lambda s: re.sub(r'[^a-z0-9\s\'\"\[\]\,-]', '', s))
        self.recipes_df[column] = self.recipes_df[column].apply(lambda s: s if not re.search(r"less_than|greater_than|sql", s) else None)
        self.recipes_df[column] = self.recipes_df[column].apply(extract_list_strings)

    def preprocess_string_column(
        self,
        column: str
    ):
        """
        Preprocess a string column in the recipes.
        """
        if column not in self.issues['nan']:
            self.issues['nan'][column] = self.recipes_df[column].isnull().sum()
        self.recipes_df[column] = self.recipes_df[column].fillna('')
        self.recipes_df[column] = self.recipes_df[column].astype(str)
        self.recipes_df[column] = self.recipes_df[column].str.lower(
        ).str.strip()

    def preprocess_int_column(
        self,
        column: str
    ):
        """
        Preprocess an integer column in the recipes.
        """
        if column not in self.issues['nan']:
            self.issues['nan'][column] = self.recipes_df[column].isnull().sum()
        self.recipes_df[column] = self.recipes_df[column].fillna(-1)
        self.recipes_df[column] = self.recipes_df[column].astype(int)

    def preprocess_date_column(
        self,
        column: str
    ):
        """
        Preprocess a date column in the recipes.
        """
        if column not in self.issues['nan']:
            self.issues['nan'][column] = self.recipes_df[column].isnull().sum()
        # Convert to datetime, extract date, handle errors
        self.recipes_df[column] = pd.to_datetime(
            self.recipes_df[column], errors='coerce')
        self.recipes_df[column] = self.recipes_df[column].dt.date
        self.recipes_df[column] = self.recipes_df[column].fillna(pd.NaT)

    def preprocess(self):
        """
        Preprocess the recipes DataFrame by cleaning specific columns.
        """

        if 'nan' not in self.issues:
            self.issues['nan'] = {}

        self.logger.info("Starting preprocessing of recipes DataFrame...")
        self.preprocess_string_column('name')
        self.preprocess_string_column('description')

        self.preprocess_list_strings_column('tags')
        self.preprocess_list_strings_column('ingredients')
        self.preprocess_list_strings_column('steps')

        self.preprocess_int_column('minutes')
        self.preprocess_int_column('contributor_id')
        self.preprocess_int_column('n_steps')
        self.preprocess_int_column('n_ingredients')

        self.preprocess_date_column('submitted')
        self.logger.info("Preprocessing completed.")

        # ------------------------------------------------------------------------------

        assert not self.recipes_df['name'].isnull().any(
        ), "Null values found in 'name' column after preprocessing."
        assert not self.recipes_df['description'].isnull().any(
        ), "Null values found in 'description' column after preprocessing."
        assert not self.recipes_df['tags'].isnull().any(
        ), "Null values found in 'tags' column after preprocessing."
        assert not self.recipes_df['ingredients'].isnull().any(
        ), "Null values found in 'ingredients' column after preprocessing."
        assert not self.recipes_df['steps'].isnull().any(
        ), "Null values found in 'steps' column after preprocessing."
        assert not self.recipes_df['minutes'].isnull().any(
        ), "Null values found in 'minutes' column after preprocessing."
        assert not self.recipes_df['contributor_id'].isnull().any(
        ), "Null values found in 'contributor_id' column after preprocessing."
        assert not self.recipes_df['n_steps'].isnull().any(
        ), "Null values found in 'n_steps' column after preprocessing."
        assert not self.recipes_df['n_ingredients'].isnull().any(
        ), "Null values found in 'n_ingredients' column after preprocessing."

        # ------------------------------------------------------------------------------

        # TODO: The number of steps in 'n_steps' does not match the length of 'steps' list
        # assert (self.recipes_df['n_steps'] == self.recipes_df['steps'].apply(len)).all(), "'n_steps' does not match the length of 'steps' after preprocessing."
        assert (self.recipes_df['n_ingredients'] == self.recipes_df['ingredients'].apply(len)).all(), "'n_ingredients' does not match the length of 'ingredients' after preprocessing."
