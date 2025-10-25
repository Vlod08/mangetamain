

import json
from typing import List, Dict
import pandas as pd
import nltk
from utils.string_utils import stop_words, fuzzy_fetch

class CountryHandler:
    def __init__(self, path: str, verbose: bool = False):
        nltk.download('stopwords')
        self.stop_words: set = stop_words
        self.countries_dataset: pd.DataFrame = None
        self.ref_countries: List[List] = None
        self.ref_regions: List[str] = None

        self.load_countries(path)

    def extract_ref_countries(self, countries_data: dict):
        """Extract relevant country information from the dataset and create a DataFrame.
        Args:
            countries_data (dict): The JSON data containing country information.
        """
        if self.verbose:
            print("Extracting reference countries and regions...")
        # Extract relevant country information from the dataset.
        ref_countries = []
        # Create a set to hold unique regions.
        ref_regions = set()
        # Create a list to hold countries with flags.
        ref_countries_with_flags = []
        for country_dict in countries_data:
            # Extract relevant fields.
            common_name = country_dict.get("name", {}).get("common", "")
            official_name = country_dict.get("name", {}).get("official", "")
            region = country_dict.get("region", "")
            subregion = country_dict.get("subregion", country_dict.get("region", ""))
            demonym = country_dict.get("demonyms", {}).get("eng", {}).get("f", "")
            flag = country_dict.get("flag", None)
            # lowercase all names for consistency
            country_info = [
                    common_name.lower(),
                    official_name.lower(),
                    demonym.lower(),
            ]
            ref_countries.append(country_info)
            ref_regions.add(region)
            ref_countries_with_flags.append([
                common_name.lower(), common_name, official_name, region, subregion, demonym, flag
            ])
        # Create a DataFrame for countries with flags.
        self.countries_dataset = pd.DataFrame(ref_countries_with_flags, columns=[
            "name", "common_name", "official_name", "region", "subregion", "demonym", "flag"
        ])
        self.ref_countries = ref_countries
        self.ref_regions = list(ref_regions)
        if self.verbose:
            print("Reference countries and regions extracted.")

    def load_countries(self, path: str):
        """Load country names from a JSON file.

        Args:
            path (str): Path to the JSON file containing country names.
        """
        try:
            if self.verbose:
                print(f"Loading countries from {path}...")
            countries_dataset = json.load(open(path, "r", encoding="utf-8"))
            if self.verbose:
                print(f"Loaded {len(countries_dataset)} countries.")
            self.extract_ref_countries(countries_dataset)
        except Exception as e:
            print(f"Error loading countries: {e}")
            return []

    def fetch_country_names_from_df(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fetch best matching country names for the given queries.

        Args:
            queries (List[str]|pd.Series): List or Series of country names to be matched.
            threshold (int, optional): Similarity threshold for matching. Defaults to 80.

        Returns:
            Dict[str, str]: A dictionary mapping query names to their best matching country names.
        """
        if self.verbose:
            print("Fetching country names from DataFrame...")
        ref_names = {
            "country": [self.ref_countries],
            "region": [self.ref_regions]
        }
        ref_cols = ["country", "region"]
        for col_idx, col in enumerate(columns):
            if self.verbose:
                print(f"\nProcessing column: {col} for country matching...")
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            for ref_col in ref_cols:
                if self.verbose:
                    print(f"  Matching for reference column: {ref_col}...")
                # Only update rows where the reference column is empty
                if col_idx > 0:
                    indexes_to_update = df[df[ref_col] == ''].index
                else:
                    indexes_to_update = df.index
                df.loc[indexes_to_update, ref_col] = df.loc[indexes_to_update, col].map(
                    lambda x: fuzzy_fetch(x, ref_names[ref_col])
                )
        return df