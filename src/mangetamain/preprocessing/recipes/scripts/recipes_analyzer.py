import logging
from pathlib import Path
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.recipes_df = self.recipes_preprocessor.recipes_df
        self.country_handler = CountryHandler(
            self.recipes_preprocessor.COUNTRIES_FILE_PATH)
        self.season_handler = SeasonHandler()

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
        fetch_columns = ["tags", "name", "description"]
        self.recipes_df = self.country_handler.fetch(
            self.recipes_df, columns=fetch_columns)
        self.logger.info("Country names fetched and updated in DataFrame.")

        # Fetch season and event labels from relevant colmns
        self.recipes_df = self.season_handler.fetch(
            self.recipes_df,
            columns=fetch_columns)
        self.logger.info("Season names fetched and updated in DataFrame.")

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

    # Analyze signature ingredients per country
    def get_signatures_countries(self, df: pd.DataFrame, top_n: int = 10) -> dict:
        """Compute country-specific signature ingredients using TF-IDF.

        This method treats each country as a "document" composed of all its recipes'
        ingredients (tokenized already as lists). It then fits a TF-IDF model and
        returns the top-N ingredients with the highest TF-IDF scores per country.

        Args:
            df: Input DataFrame expected to contain columns 'country' and 'ingredients'.
                - 'country' should be a categorical/str label per recipe.
                - 'ingredients' should be a list of tokens per recipe (not raw strings).
            top_n: Number of top TF-IDF ingredients to return per country.

        Returns:
            A dict mapping country -> list of top-N signature ingredient tokens.
        """
        df = df[df['country'] != '']

        # Aggregate all ingredient lists per country into a single list per country
        country_docs_lists = df.groupby('country')['ingredients'].sum()

        # Configure vectorizer to accept pre-tokenized input (lists)
        vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x,
                                     lowercase=False, max_df=0.5, max_features=14000)

        # Fit TF-IDF on the per-country documents (each document is a list of tokens)
        tfidf_matrix = vectorizer.fit_transform(country_docs_lists)

        ingredients = vectorizer.get_feature_names_out()

        # Build a convenient DataFrame: rows=countries, cols=ingredients, values=tf-idf
        df_tfidf = pd.DataFrame(
            tfidf_matrix.toarray(), index=country_docs_lists.index, columns=ingredients
        )

        signatures = {}
        for country in df_tfidf.index:
            # Pick the "top_n" highest-scoring ingredients for this country
            top_scores_series = df_tfidf.loc[country].nlargest(top_n)
            signatures[country] = top_scores_series.to_dict()

        return signatures

    def get_signatures_seasons(self, df: pd.DataFrame, top_n: int = 10) -> dict:
        """Compute season-specific signature ingredients using TF-IDF.

        Similar to the country signatures, this method groups recipes by their
        inferred 'season' label, aggregates ingredient tokens per season, fits a
        TF-IDF model, and returns the top-N ingredients per season.

        Args:
            df: Input DataFrame with columns 'season' and 'ingredients'.
                - 'season' should contain season labels like 'winter', 'spring', etc.
                - 'ingredients' should be a list of tokens per recipe.
            top_n: Number of top TF-IDF ingredients to return per season.

        Returns:
            A dict mapping season -> list of top-N signature ingredient tokens.
        """
        df = df[df['season'] != '']

        # Keep only rows with valid season labels
        saisons_valides = ['winter', 'spring', 'summer', 'fall']
        df_clean = df[df['season'].isin(saisons_valides)]

        # Aggregate ingredients per season into a single list per season
        season_docs_lists = df_clean.groupby('season')['ingredients'].sum()

        # Vectorizer configured for pre-tokenized input (lists)
        vectorizer = TfidfVectorizer(
            preprocessor=lambda x: x, tokenizer=lambda x: x, lowercase=False)

        # Fit TF-IDF per season document
        tfidf_matrix = vectorizer.fit_transform(season_docs_lists)

        ingredients = vectorizer.get_feature_names_out()

        # DataFrame of TF-IDF scores: rows=seasons, cols=ingredients
        df_tfidf = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=season_docs_lists.index,
            columns=ingredients
        )

        signatures = {}
        for season in df_tfidf.index:
            # Top-N ingredients with highest TF-IDF for this season
            top_scores_series = df_tfidf.loc[season].nlargest(top_n)
            signatures[season] = top_scores_series.to_dict()

        return signatures


if __name__ == "__main__":
    analyzer = RecipesAnalyzer()
    analyzer.analyze()
    analyzer.export_results(dir_path="output/recipes_analysis")
    print()
    print(analyzer.recipes_df[["name", "tags", "description",
          "country", "region", "season", "event"]].head(10))
