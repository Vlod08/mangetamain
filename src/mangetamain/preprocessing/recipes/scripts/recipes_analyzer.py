import pandas as pd
import sys

from recipes_preprocessing import RecipesPreprocessor
from country_handler import CountryHandler
from seasonnality_handler import SeasonHandler


class RecipesAnalyzer:

    info = {}

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            print("Initializing RecipesAnalyzer...")
        self.recipes_preprocessor = RecipesPreprocessor()
        if self.verbose:
            print("RecipesPreprocessor initialized.")
        self.country_handler = CountryHandler(
            self.recipes_preprocessor.COUNTRIES_FILE_PATH, verbose=self.verbose)
        if self.verbose:
            print("CountryHandler initialized.")
        self.season_handler = SeasonHandler(path='', verbose=self.verbose)
        if self.verbose:
            print("SeasonHandler initialized.")
        self.recipes_df = self.recipes_preprocessor.recipes_df

    def analyze(self):
        """
        Analyze the recipes DataFrame for inconsistencies and missing values.
        """

        self.info["total_recipes"] = {
            "name": "Total Recipes",
            "value": len(self.recipes_df)
        }
        self.info["nb_columns"] = {
            "name": "Number of Columns",
            "value": len(self.recipes_df.columns)
        }
        self.info["columns"] = {
            "name": "Columns",
            "value": list(self.recipes_df.columns)
        }

        # Fetch country names from relevant columns
        columns_for_country_matching = ["tags"]  # , "name", "description"]
        self.recipes_df = self.country_handler.fetch_country_names_from_df(
            self.recipes_df, columns=columns_for_country_matching)
        if self.verbose:
            print("\n\nCountry names fetched and updated in DataFrame.")
            print(self.recipes_df[["country", "region"]])

        # Fetch season and event labels
        self.recipes_df = self.season_handler.get_periods_all(self.recipes_df)
        if self.verbose:
            print("\n\nSeason and event labels fetched and updated in DataFrame.")
            print(self.recipes_df[["season", "event"]])


if __name__ == "__main__":
    if sys.argv and len(sys.argv) > 1:
        verbose = int(sys.argv[1]) != 0
        print(f"Verbose mode: {verbose}")
    else:
        verbose = False
    analyzer = RecipesAnalyzer(verbose)
    analyzer.analyze()
    print("\n")
    print(analyzer.info)
