# core/handlers/country_handler.py
from __future__ import annotations
import json
from typing import List
import pandas as pd
from dataclasses import dataclass

from core.handlers.recipes_handler import RecipesHandler

@dataclass
class CountryHandler(RecipesHandler):
    """Class for handling country-related data processing.
    Inherits from RecipesHandler.
    """

    def __post_init__(self, path: str = None):
        """Initialize the CountryHandler with the given path.
        Args:
            path (str): Path to the countries JSON file.
        """
        super().__post_init__()
        self.logger.info("CountryHandler initialized.")

    def extract_ref_countries(self, countries_data: dict):
        """Extract reference countries and regions from the provided JSON data.
        Args:
            countries_data (dict): The JSON data containing country information.
        """
        self.logger.info("Extracting reference countries and regions...")

        # Extract relevant country information from the dataset.
        ref_countries = []
        ref_demonyms = []
        # Create a set to hold unique regions.
        ref_regions = set()
        # Create a list to hold ref data.
        ref_dataset = []

        for country_dict in countries_data:
            # Extract relevant fields.
            common_name = country_dict.get("name", {}).get("common", "")
            official_name = country_dict.get("name", {}).get("official", "")
            region = country_dict.get("region", "")
            subregion = country_dict.get("subregion", country_dict.get("region", ""))
            demonym = country_dict.get("demonyms", {}).get("eng", {}).get("f", "")
            flag = country_dict.get("flag", None)

            # lowercase all names for consistency
            ref_countries.append(common_name.lower())
            ref_demonyms.append(demonym.lower())
            ref_regions.add(region.lower())

            ref_dataset.append([
                common_name.lower(), common_name, official_name, region, subregion, demonym, flag
            ])
        
        # Set the values of class attributes
        self.ref_dataset = pd.DataFrame(ref_dataset, columns=[
            "name", "common_name", "official_name", "region", "subregion", "demonym", "flag"
        ])
        self.ref_names = {
            "country": [ref_countries, ref_demonyms],
            "region": [list(ref_regions)]
        }
        self.logger.info("Reference countries and regions extracted.")

    def build_ref(self, path: str = None):
        """Build the reference dataset from the JSON file.

        Args:
            path (str): Path to the JSON file containing country names.
        """
        if not path:
            if self.ref_path:
                path = self.ref_path
            else:
                raise ValueError("Path to countries JSON file must be provided.")
        try:
            self.logger.info(f"Loading countries from {path}...")
            countries_dataset = json.load(open(path, "r", encoding="utf-8"))
            self.logger.info(f"Loaded {len(countries_dataset)} countries.")
            self.extract_ref_countries(countries_dataset)
        except Exception as e:
            self.logger.error(f"Error loading countries: {e}")
            return []

    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer region names from country names in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The updated DataFrame with inferred region names.
        """
        if self.ref_dataset is None:
            self.logger.warning("Countries dataset is not loaded.")
            return pd.Series(dtype=str)
    
        # Infer regions from countries if region is still missing
        missing_regions = df['region'] == ''
        if missing_regions.any():
            self.logger.info("Inferring regions from country names for missing regions...")
            country_to_region = dict(zip(
                self.ref_dataset['name'], 
                self.ref_dataset['region']
            ))

            df.loc[missing_regions, 'region'] = df.loc[missing_regions, 'country'].map(country_to_region).fillna('')

        return df
    
    def fetch(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fetch country names from the specified columns in the DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame to process.
            columns (List[str]): List of column names to search for country names.
        Returns:
            pd.DataFrame: The updated DataFrame with country names fetched.
        """
        self.logger.info("Fetching country names from DataFrame...")
        df = super().fetch(df, columns)
        self.logger.info("Country names fetching completed.")
        return df
