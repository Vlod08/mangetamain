
from typing import Iterable, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import ast
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer
import re
from functools import lru_cache

# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords', quiet=True)
# STOP_WORDS_EN = set(stopwords.words('english'))


def is_list_string(s: str) -> bool:
    """Check if a string is a list of strings representation."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return False
    try:
        lst = ast.literal_eval(s)
        return all(isinstance(x, str) for x in lst)
    except Exception:
        return False

def is_list_floats_string(s: str) -> bool:
    """Check if a string is a list of floats representation."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return False
    try:
        lst = ast.literal_eval(s)
        return all(isinstance(x, float) or x is None for x in lst)
    except Exception:
        return False

def extract_list_strings(str_l: str) -> List[str]:
    """Extracts a list of strings from a string representation of a list.

    Returns:
        list: A list of strings extracted from the input string.
    """
    if pd.isna(str_l) or str_l == '':
        return []
    try:
        list_strings = ast.literal_eval(str_l)
        if not isinstance(list_strings, list):
            return []
        # Remove empty or whitespace-only strings
        cleaned = [x.strip() for x in list_strings if isinstance(x, str) and x.strip() != '']
        return cleaned
    except (ValueError, SyntaxError):
        # If literal_eval fails, return empty list
        return []
    
def extract_list_floats(str_l: str) -> List[float]:
    """Extracts a list of floats from a string representation of a list.

    Returns:
        list: A list of floats extracted from the input string.
    """
    if pd.isna(str_l) or str_l == '':
        return []
    try:
        list_floats = ast.literal_eval(str_l)
        if not isinstance(list_floats, list):
            return []
        # Convert to floats and remove non-convertible entries
        cleaned = []
        for x in list_floats:
            try:
                cleaned.append(float(x))
            except (ValueError, TypeError):
                continue
        return cleaned
    except (ValueError, SyntaxError):
        # If literal_eval fails, return empty list
        return []

# Covers ISO, European, US, textual, and timestamp-like formats
DATETIME_PATTERN = re.compile(
    r"""
    ^(?:                                   # entire string must match
        # ISO-like: 2024-03-12, 2024/03/12, 2024.03.12
        \d{4}[-/.]\d{1,2}[-/.]\d{1,2}
        |
        # European/US-like: 12/03/2024 or 03-12-24
        \d{1,2}[-/]\d{1,2}[-/]\d{2,4}
        |
        # Textual month: 12 Mar 2024, March 12, 2024
        (?:\d{1,2}\s*[A-Za-z]{3,9}|[A-Za-z]{3,9}\s*\d{1,2})(?:,?\s*\d{2,4})?
    )$                                     # must end here
    """,
    re.VERBOSE,
)

def looks_like_datetime(s: str) -> bool:
    """Check if a string has potential to represent a date or datetime."""
    if s.startswith("[") and s.endswith("]"):
        return False
    if (not isinstance(s, str)) or (len(s) < 4) or (len(s) > 20):
        return False
    return bool(DATETIME_PATTERN.search(s))

def mean_token_scorer(s1, s2, **kwargs):
    """Custom scorer = mean of token_set_ratio and token_sort_ratio."""
    score1 = fuzz.token_set_ratio(s1, s2)
    score2 = fuzz.token_sort_ratio(s1, s2)
    return (score1 + score2) / 2

def fuzzy_fetch(query: str, list_ref_names: List[List[str]], threshold: int = 80):
    """
    Fetch the best-matching reference name across several semantically related groups
    (e.g., countries and demonyms), using fuzzy string matching.

    The function returns the reference name (from the first group) corresponding
    to the best-matching string among all groups if the similarity score is above
    the threshold. Otherwise, it returns an empty string.

    Args:
        queries (str): The input string to be matched.
        list_ref_names (List[List[str]]): A list of groups of reference names.
            Each group is a list of strings of equal length. For example:
                [
                    ["france", "spain", "morocco"],      # group 0: countries
                    ["french", "spanish", "moroccan"]    # group 1: demonyms
                ]
        threshold (int, optional): Minimum similarity score to consider a match valid.
            Defaults to 80.

    Returns:
        str: The best-matching reference name (from the first group), or an empty string
            if no match above threshold is found.

    Example:
        >>> queries = ["french", "italy"]
        >>> list_ref_names = [
        ...     ["france", "spain", "morocco", "italy"],
        ...     ["french", "spanish", "moroccan", "italian"]
        ... ]
        >>> fuzzy_fetch(queries, list_ref_names, threshold=80)
        'france'
    """
    if not query:
        return ''
    if isinstance(query, list):
        query = ' '.join(query)

    # clean query
    query_clean = query.replace('-', ' ')

    # Flatten only once outside in your class
    flat_refs = [ref for group in list_ref_names for ref in group]
    nb_refs_per_group = len(list_ref_names[0])

    scores = process.extractOne(query_clean, flat_refs, scorer=mean_token_scorer)
    if not scores or scores[1] < threshold:
        return '', scores[1]

    # Find the index of the best match in the first group
    best_match_idx = scores[2] % nb_refs_per_group
    # TODO: special case for "american" demonym
    if scores[0]=="american":
        return "united states", scores[1]
    return list_ref_names[0][best_match_idx], scores[1]

def contains_any(values: Iterable[str], items: Iterable[str]) -> bool:
    """Check if any of the items are present in the values."""
    s = set(v.strip().lower() for v in values if v)
    return any(t in s for t in items)

def extract_classes(list_strings: List[str|List[str]]) -> List[str]:
    """Extracts unique classes from a list of strings or list of strings.
    Args:
        list_strings (List[str|List[str]]): A list containing strings or lists of strings.
    Returns:
        List[str]: A list of unique classes extracted from the input.
    """
    mlb = MultiLabelBinarizer()
    # Fit the MultiLabelBinarizer on the list of strings
    mlb.fit(list_strings)
    return mlb.classes_.tolist()
