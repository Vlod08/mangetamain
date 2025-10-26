import logging
from pathlib import Path
import pandas as pd
import json

from recipes_preprocessing import RecipesPreprocessor
from country_handler import CountryHandler
from seasonnality_handler import SeasonHandler

from utils.utils import setup_logging

class RecipesAnalyzer:

    info = {}

    def __init__(self):
        # Setup logging
        log_file = setup_logging(log_level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Logging initialized. Writing to {log_file}")

        self.logger.info("Initializing RecipesAnalyzer...")
        self.recipes_preprocessor = RecipesPreprocessor()
        self.recipes_preprocessor.preprocess()
        self.country_handler = CountryHandler(self.recipes_preprocessor.COUNTRIES_FILE_PATH)
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
        columns_for_country_matching = ["tags", "name", "description"]
        self.recipes_df = self.country_handler.fetch(
            self.recipes_df, columns=columns_for_country_matching)
        self.logger.info("Country names fetched and updated in DataFrame.")

    def export_results(self, dir_path: str):
        """
        Export the analysis results to the specified directory.
        Args:
            dir_path (str): Path to export the results.
        """
        self.logger.info(f"Exporting results to {dir_path}...")
        dir_path = Path(dir_path)

        # Create the directory if it doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)

        # Export the recipes DataFrame to CSV
        dataset_path = dir_path / "recipes.csv"
        self.recipes_df.to_csv(dataset_path, index=False)

        # Export the analysis info to a json file
        with open(f"{dir_path}/recipes_info.json", "w", encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent=4)
        
        self.logger.info("Results exported successfully.")

        # Fetch season and event labels
        self.recipes_df = self.season_handler.get_periods_all(self.recipes_df)
        if self.verbose:
            print("\n\nSeason and event labels fetched and updated in DataFrame.")
            print(self.recipes_df[["season", "event"]])


if __name__ == "__main__":
    analyzer = RecipesAnalyzer()
    analyzer.analyze()
    analyzer.export_results(dir_path="output/recipes_analysis")
    print()
    print(analyzer.recipes_df[["name", "tags", "description", "country", "region"]].head(10))
