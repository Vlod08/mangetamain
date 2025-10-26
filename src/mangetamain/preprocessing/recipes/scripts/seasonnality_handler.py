from typing import List
import pandas as pd
import re


class SeasonHandler:
    def __init__(self, path: str, verbose: bool = False):
        """Initialize the SeasonHandler.

        Args:
            path: Path where season/event references could be loaded from.
            verbose: If True, enables extra logging (reserved for future use).

        Side effects:
            - Sets ``self.stop_words`` from utils.string_utils.stop_words.
            - Initializes ``self.ref_season`` and ``self.ref_event`` by calling
              :meth:`load_dataset`.
        """
        # Domain stop words possibly used by future fuzzy matching helpers
        self.stop_words: set = None
        # References that will store valid season and event labels
        self.ref_season: List[str] = None
        self.ref_event: List[str] = None

        # Load (or initialize) reference vocabularies
        self.load_dataset(path)

    def load_dataset(self, path: str) -> None:
        """Load reference period vocabularies.

        For now, this method initializes in-memory lists of seasons and events.
        The ``path`` parameter is reserved for future versions where references could
        be loaded from a CSV or other source.

        Args:
            path: Path to a potential reference file (currently unused).

        Returns:
            None
        """
        # Static references; adjust or replace with file-based loading if needed
        self.ref_season = ['winter', 'spring', 'summer', 'fall']
        self.ref_event = ['christmas', 'halloween', 'thanksgiving']

    def get_period_type_column(self, column: pd.Series, period_type: List[str]) -> pd.Series:
        """Extract occurrences of a given period type from a text Series.

        This scans each string in ``column`` and finds all occurrences matching any
        value in ``period_type`` (using a regex that escapes special characters).
        Duplicates per row are removed and the remaining matches are returned as a list.

        Args:
            column: Pandas Series of strings (e.g., recipe names, tags, descriptions).
            period_type: List of period tokens to search for (e.g., seasons or events).

        Returns:
            A Series of lists, where each element contains the unique matched periods
            for the corresponding row. Empty lists indicate no match.
        """
        # Build a safe regex that matches any period label as a literal string
        escaped_periods = [re.escape(period) for period in period_type]
        pattern = '|'.join(escaped_periods)

        # Find all matches per row (returns list of matches, possibly with duplicates)
        all_matches = column.str.findall(pattern)

        # Convert to unique lists to avoid returning the same period multiple times
        # periods_column = all_matches.apply(set).apply(list)
        return all_matches

    def get_prioritized_period(self, df: pd.DataFrame) -> pd.Series:
        """Resolve and prioritize a single period per row from multiple sources.

        The input DataFrame is expected to contain three columns: ``tags``, ``name``,
        and ``description``; each cell should be a list of candidate periods.
        This method picks the first element from each source (if present) and prioritizes
        them in the following order: tags > name > description.

        Args:
            df: DataFrame with columns ``tags``, ``name``, ``description`` where each
                value is a list (possibly empty) of detected periods.

        Returns:
            A Series of strings containing the chosen period for each row. Empty string
            indicates that no period could be selected.
        """
        # Take the first candidate of each column when available
        tags_period = df['tags'].str[0]
        name_period = df['name'].str[0]
        description_period = df['description'].str[0]

        # Priority: tags first, then name, then description
        final_seasons = tags_period.combine_first(
            name_period).combine_first(description_period)
        # Ensure missing values are represented as empty strings for downstream code
        final_seasons = final_seasons.fillna('')
        return final_seasons

    def get_periods_all(self, recipe: pd.DataFrame) -> pd.DataFrame:
        """Compute season and event labels for all recipes.

        For each of the text fields (``tags``, ``name``, ``description``), this method
        extracts candidate seasons and events and then reduces them to a single label
        per row using a prioritization strategy.

        Args:
            recipe: Input DataFrame that must include the string columns
                ``tags``, ``name``, and ``description``.

        Returns:
            A copy of the input DataFrame with two additional columns:
            - ``season``: chosen season label per row (or empty string)
            - ``event``: chosen event label per row (or empty string)
        """
        # Work on a copy to avoid mutating the caller's DataFrame
        recipe_final = recipe.copy()

        # Extract season candidates from each textual field
        tags = self.get_period_type_column(recipe['tags'], self.ref_season)
        name = self.get_period_type_column(recipe['name'], self.ref_season)
        description = self.get_period_type_column(
            recipe['description'], self.ref_season)

        # Decide a single season based on ambiguity-minimizing heuristic (see choose_period)
        df_seasons = pd.DataFrame(
            {'tags': tags, 'name': name, 'description': description})
        seasons = self.get_prioritized_period(df_seasons)

        # Extract event candidates; note: when using Series.apply with an extra argument,
        # you must pass it through the "args" parameter. Here we use the vectorized version
        # for consistency with seasons above.
        tags = self.get_period_type_column(recipe['tags'], self.ref_event)
        name = self.get_period_type_column(recipe['name'], self.ref_event)
        description = self.get_period_type_column(
            recipe['description'], self.ref_event)

        df_events = pd.DataFrame(
            {'tags': tags, 'name': name, 'description': description})
        # Reuse the same selection rule; if you intend a different rule, implement it here
        events = self.get_prioritized_period(df_events)

        # Attach results to the returned DataFrame
        recipe_final['season'] = seasons
        recipe_final['event'] = events

        return recipe_final
