
from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import ast
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer

# from nltk.corpus import stopwords
# nltk.download('stopwords')
# STOP_WORDS_EN = set(stopwords.words('english'))

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

def fuzzy_fetch(
    queries: Union[str, List[str]],
    list_ref_names: List[List[str]],
    threshold: int = 80
) -> str:
    """
    Fetch the best-matching reference name across several semantically related groups
    (e.g., countries and demonyms), using fuzzy string matching.

    The function returns the reference name (from the first group) corresponding
    to the best-matching string among all groups if the similarity score is above
    the threshold. Otherwise, it returns an empty string.

    Args:
        queries (Union[str, List[str]]): The input string or list of strings to be matched.
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
    # ---- 1. Input normalization ----
    if not queries:
        return ''
    if isinstance(queries, str):
        queries = [queries]

    # Validate groups
    num_groups = len(list_ref_names)
    if num_groups == 0 or any(len(lst) != len(list_ref_names[0]) for lst in list_ref_names):
        raise ValueError("All groups in list_ref_names must have the same length and be non-empty.")

    n_refs = len(list_ref_names[0])

    # ---- 2. Flatten all reference groups for batch fuzzy matching ----
    flat_refs = [ref for group in list_ref_names for ref in group]

    # ---- 3. Compute similarity scores ----
    # Shape before reshape: (len(queries), num_groups * n_refs)
    scores = process.cdist(
        queries,
        flat_refs,
        scorer=fuzz.ratio,
        workers=-1
    )

    # ---- 4. Reshape to (num_groups, len(queries), n_refs) ----
    scores = np.reshape(scores, (len(queries), num_groups, n_refs))
    scores = np.transpose(scores, (1, 0, 2))  # (num_groups, len(queries), n_refs)

    # ---- 5. Get best match per group ----
    # max across queries → best score per ref name
    best_score_per_ref = np.max(scores, axis=1)  # (num_groups, n_refs)
    # then best match within each group
    best_score_per_group = np.max(best_score_per_ref, axis=1)  # (num_groups,)
    best_idx_per_group = np.argmax(best_score_per_ref, axis=1)  # (num_groups,)

    # ---- 6. Get global best across groups ----
    global_group_idx = np.argmax(best_score_per_group)
    global_best_score = best_score_per_group[global_group_idx]
    global_best_idx = best_idx_per_group[global_group_idx]

    # ---- 7. Threshold filtering ----
    if global_best_score < threshold:
        return ''

    # ---- 8. Always return from the first group ----
    return list_ref_names[0][global_best_idx]

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