
from typing import List, Tuple, Dict
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def fuzzy_fetch(queries: List[str], list_ref_names: List[List[str]], threshold: int = 80) -> str:
    """Fetches best matching strings for each column based on the provided dictionary.
    Args:
        queries (List[str]): The input list of strings to be matched.
        list_ref_names (List[List[str]]): A list of lists of reference names for matching.
            The order of lists matters: the first list with a match above the threshold is used.
        threshold (int, optional): The minimum similarity score to consider a match valid. Defaults to 80.
    Returns:
        str: The best matching string if found, otherwise an empty string.

    Example:
        queries = ["frnace", "germnay"]
        list_ref_names = [["france", "germany", "italy"], ["spain", "portugal"]]
        result = fuzzy_fetch(queries, list_ref_names, threshold=80)
        # result would be "france" since it matches "frnace" with a score above the threshold.
    """
    if not queries:
        return ''

    for ref_names in list_ref_names:
        # Compute similarity matrix (len(queries) x len(choices))
        scores = process.cdist(queries, ref_names, scorer=fuzz.ratio)

        # Find best match for all queries
        best_idx_per_query = np.argmax(scores, axis=1)
        best_scores_per_query = scores[np.arange(len(queries)), best_idx_per_query]
        best_score = np.max(best_scores_per_query)
        best_idx = best_idx_per_query[np.argmax(best_scores_per_query)]
        if best_score >= threshold:
            return ref_names[best_idx]  # Stop at first found match for this column

    return ''

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