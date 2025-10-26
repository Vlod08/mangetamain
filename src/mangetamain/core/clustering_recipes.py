"""Compute recipe similarity using time-related tags.

This module re-implements the logic used in `similarity.ipynb` to extract
time-related tags from the `tags` column of the recipes dataset and build a
cosine-similarity matrix. It reuses `RecipesDataset` to load recipes, and
exposes a small API suitable for other code to call.

Public functions
 - extract_time_tags(df) -> DataFrame with columns ['id','time_tags','time_tags_str']
 - compute_time_tag_similarity(df, sample_n=None, random_state=1) -> pd.DataFrame (similarity)
 - get_similar_recipes(sim_df, recipe_id, top_n=5) -> pd.Series

The implementation follows the notebook's approach: parse tag strings, select
time-related tags with regex, vectorize with CountVectorizer and compute
cosine similarity.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import re
import ast
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.dataset import RecipesDataset


# --- Combined features (time-tags + ingredients) similarity ---
@st.cache_data(show_spinner=False)
def ingred_tokenizer(ings: object) -> list:
	"""Normalize an ingredients value into a list of tokens.

	Accepts already-parsed lists, or string representations (will try ast.literal_eval).
	Returns a list of lowercase tokens.
	"""
	if isinstance(ings, list):
		return [str(t).strip().lower() for t in ings if t]
	if pd.isna(ings):
		return []
	if isinstance(ings, str):
		try:
			val = ast.literal_eval(ings)
			if isinstance(val, (list, tuple)):
				return [str(t).strip().lower() for t in val if t]
		except Exception:
			# fallback: split the string on whitespace
			return [tok.strip().lower() for tok in ings.split() if tok.strip()]
	return []


def merge_temporal_ingredient_features(row: pd.Series) -> str:
	"""Create a single space-separated feature string from time-tags and ingredients tokens."""
	time_bits = row.get("time_tags") or []
	if isinstance(time_bits, (list, tuple)):
		time_str = " ".join([str(t) for t in time_bits if t])
	else:
		time_str = str(time_bits)
	ing_tokens = row.get("_proc_ings") or []
	ing_str = " ".join(ing_tokens)
	return f"{time_str} {ing_str}".strip()


@st.cache_data(show_spinner=False)
def vectorize_and_score(df: pd.DataFrame, sample_n: int | None = None, random_state: int = 42) -> pd.DataFrame:
	"""Build a CountVectorizer on combined time+ingredient features and return cosine similarity DataFrame.

	The function will create a temporary column `_proc_ings` with tokenized ingredients
	and a column `_merged_feats` with the combined string used for vectorization.
	"""
	# operate on a copy to avoid mutating caller's DataFrame
	src = df.copy()

	if sample_n is not None and sample_n > 0 and len(src) > sample_n:
		src = src.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

	# Tokenize ingredients
	src["_proc_ings"] = src.get("ingredients", pd.Series([[]] * len(src))).apply(ingred_tokenizer)

	# Ensure time_tags exist (try to extract if necessary)
	if "time_tags" not in src.columns:
		tt = extract_time_tags(src)
		src = src.merge(tt[["id", "time_tags"]], on="id", how="left")

	# Merge features into a single string
	src["_merged_feats"] = src.apply(merge_temporal_ingredient_features, axis=1)

	# If there are no tokens, return empty
	if src["_merged_feats"].astype(bool).sum() == 0:
		return pd.DataFrame()

	vect = CountVectorizer(tokenizer=lambda s: s.split(" "))
	M = vect.fit_transform(src["_merged_feats"].astype(str))

	sim = cosine_similarity(M)
	simdf = pd.DataFrame(sim, index=src["id"].values, columns=src["id"].values)
	return simdf


def fetch_similar_items(sim_df: pd.DataFrame, recipe_id, top_n: int = 25) -> pd.Series:
	"""Return top-N similar recipe ids (with scores) given a combined similarity DataFrame."""
	if sim_df.empty:
		return pd.Series(dtype=float)
	if recipe_id not in sim_df.index:
		raise KeyError(f"Recipe {recipe_id} not found in combined similarity matrix")
	scores = sim_df[recipe_id].sort_values(ascending=False)
	scores = scores.drop(labels=recipe_id, errors="ignore")
	return scores.head(top_n)



def _string_to_list(tags_string: object) -> list:
	"""Convert a tags field (possibly list or string) into a Python list.

	The RAW dataset stores tags in various formats. If the value is already a 
	list, return it; if it's a stringlike "['a']['b']" or "['a','b']", attempt 
	to clean it and ast.literal_eval it; on any failure, return an empty list.
	"""
	if isinstance(tags_string, list):
		return tags_string
	if pd.isna(tags_string):
		return []
	if not isinstance(tags_string, str):
		return []
	try:
		# Some rows contain "][" sequences between lists; replace with comma
		cleaned_string = re.sub(r"\]\[", ", ", tags_string.strip())
		# Ensure it looks like a list literal
		if not (cleaned_string.startswith("[") and cleaned_string.endswith("]")):
			cleaned_string = f'[{cleaned_string.strip("[]")}]'
		return ast.literal_eval(cleaned_string)
	except Exception:
		# Fallback: try splitting on commas
		try:
			return [t.strip() for t in re.split(r",\s*", tags_string.strip("[]")) if t.strip()]
		except Exception:
			return []


def get_time_tags(tag_list: Iterable[str]) -> list:
	"""Return the list of time-related tags from a tag list.

	Matches patterns like '15-minutes', '4-hours', '1-day-or-more',
	'30-minutes-or-less', etc.
	"""
	time_patterns = [
		r"\b\d+-minutes?\b",
		r"\b\d+-hours?\b",
		r"\b\d+-day(?:s)?-or-more\b",
		r"\b\d+-minutes-or-less\b",
		r"\b\d+-hours-or-less\b",
	]
	res = []
	if not tag_list:
		return res
	for t in tag_list:
		try:
			for pattern in time_patterns:
				if re.search(pattern, t):
					res.append(t)
					break
		except Exception:
			continue
	return res


@st.cache_data(show_spinner=False)
def extract_time_tags(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a dataframe with recipe id and extracted time tags.

	Output columns: ['id','time_tags','time_tags_str'] where time_tags is a list
	and time_tags_str is the space-joined representation used for vectorization.
	"""
	if "id" not in df.columns:
		raise KeyError("Input dataframe must contain an 'id' column")

	tags_series = df.get("tags", pd.Series([[]] * len(df), index=df.index))

	parsed = tags_series.apply(_string_to_list)
	time_tags = parsed.apply(get_time_tags)
	time_tags_str = time_tags.apply(lambda tags: " ".join(tags) if tags else "")

	out = pd.DataFrame({"id": df["id"].values, "time_tags": time_tags.values, "time_tags_str": time_tags_str.values})
	return out


@st.cache_data(show_spinner=False)
def compute_time_tag_similarity(df: pd.DataFrame, sample_n: int | None = None, random_state: int = 1) -> pd.DataFrame:
	"""Compute cosine similarity DataFrame between recipes based on time tags.

	Parameters
	- df: recipes dataframe (must contain 'id' and 'tags' or precomputed 'time_tags_str')
	- sample_n: if provided, randomly sample this many recipes before computing
	- random_state: RNG seed for sampling

	Returns
	- similarity_df: pandas.DataFrame indexed and columned by recipe id
	"""
	# If user already provided a precomputed 'time_tags_str', use it; otherwise extract
	if "time_tags_str" in df.columns and "id" in df.columns:
		tags_df = df[["id", "time_tags_str"]].copy()
		tags_df = tags_df.rename(columns={"time_tags_str": "time_tags_str"})
	else:
		tags_df = extract_time_tags(df)[["id", "time_tags_str"]].copy()

	if sample_n is not None and sample_n > 0 and len(tags_df) > sample_n:
		tags_df = tags_df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

	# If there are no time tags at all, return empty DataFrame
	if tags_df["time_tags_str"].astype(bool).sum() == 0:
		return pd.DataFrame()

	# Treat hyphenated words (e.g. '15-minutes') as single tokens.
	vectorizer = CountVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
	X = vectorizer.fit_transform(tags_df["time_tags_str"].astype(str))

	# If X is empty (no tokens), return empty
	if X.shape[1] == 0:
		return pd.DataFrame()

	sim = cosine_similarity(X)
	sim_df = pd.DataFrame(sim, index=tags_df["id"].values, columns=tags_df["id"].values)
	return sim_df


def get_similar_recipes(sim_df: pd.DataFrame, recipe_id, top_n: int = 5) -> pd.Series:
	"""Return top-n similar recipes (scores) for a given recipe id from sim_df.

	If recipe_id is not found, KeyError is raised.
	"""
	if sim_df.empty:
		return pd.Series(dtype=float)
	if recipe_id not in sim_df.index:
		raise KeyError(f"Recipe id {recipe_id} not found in similarity matrix")
	similar_scores = sim_df[recipe_id].sort_values(ascending=False)
	similar_scores = similar_scores.drop(labels=recipe_id, errors="ignore")
	return similar_scores.head(top_n)


def compute_similarity_from_dataset(anchor: Path | str, sample_n: int | None = None, random_state: int = 1) -> pd.DataFrame:
	"""Helper to load the recipes using RecipesDataset(anchor) and compute similarity.

	anchor is typically Path(__file__) from the caller; accepts str for convenience.
	"""
	anchor_path = Path(anchor) if not isinstance(anchor, Path) else anchor
	ds = RecipesDataset(anchor=anchor_path)
	df = ds.load()
	return compute_time_tag_similarity(df, sample_n=sample_n, random_state=random_state)


def save_time_similarity_matrix(
	anchor: Path | str,
	out_path: Path | str = "data/processed/time_similarity.parquet",
    sample_n: int | None = None,
    random_state: int = 1,
    as_csv: bool = False,
    ) -> Path:
    """
    Compute and save the cosine similarity matrix based on time-related tags.

    Parameters
    ----------
    anchor : Path | str
        Typically Path(__file__) or the project root (used by RecipesDataset).
    out_path : Path | str, default 'data/processed/time_similarity.parquet'
        Where to save the similarity matrix.
    sample_n : int | None
        If provided, randomly sample this many recipes before computing similarity.
    random_state : int, default 1
        Random seed for reproducible sampling.
    as_csv : bool, default False
        If True, saves the matrix as CSV instead of Parquet.

    Returns
    -------
    Path
        Path to the saved similarity file.
    """
    # Compute similarity
    sim_df = compute_similarity_from_dataset(anchor, sample_n=sample_n, random_state=random_state)

    if sim_df.empty:
        raise ValueError("Similarity matrix is empty — no time tags found in dataset.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    if as_csv:
        sim_df.to_csv(out_path, index=True)
    else:
        sim_df.to_parquet(out_path, index=True)

    print(f"Saved time-tag similarity matrix to: {out_path} ({sim_df.shape[0]}×{sim_df.shape[1]})")
    return out_path
