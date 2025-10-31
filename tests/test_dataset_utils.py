# tests/test_dataset_utils.py
from __future__ import annotations
import pandas as pd
import numpy as np

from mangetamain.core.dataset import DatasetLoader


def test_to_hashable_scalars_and_collections():
    # scalars stay the same
    assert DatasetLoader.to_hashable(1) == 1
    assert DatasetLoader.to_hashable(3.14) == 3.14
    assert DatasetLoader.to_hashable("s") == "s"

    # list -> tuple, nested preserved
    assert DatasetLoader.to_hashable([1, 2, [3, 4]]) == (1, 2, (3, 4))

    # tuple stays tuple (but elements converted)
    assert DatasetLoader.to_hashable(("a", ["b"])) == ("a", ("b",))

    # dict -> sorted tuple of (k, v)
    d = {"b": 2, "a": 1}
    t = DatasetLoader.to_hashable(d)
    assert isinstance(t, tuple)
    assert t[0][0] == "a" and t[1][0] == "b"

    # numpy array -> tuple
    arr = np.array([1, 2, 3])
    assert DatasetLoader.to_hashable(arr) == (1, 2, 3)


def test_is_non_hashable_column_and_make_hashable():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [[1], [2], [3]],
        "c": [{"x": 1}, {"x": 2}, {"x": 3}],
    })

    # 'a' is hashable-only, 'b' and 'c' are non-hashable
    # use truthy/falsy checks to handle numpy.bool_ return values
    assert not DatasetLoader.is_non_hashable_column(df["a"])
    assert DatasetLoader.is_non_hashable_column(df["b"])
    assert DatasetLoader.is_non_hashable_column(df["c"])

    hashable = DatasetLoader.make_dataframe_hashable(df)
    # columns b and c should now contain tuples
    assert all(isinstance(x, tuple) for x in hashable["b"])
    assert all(isinstance(x, tuple) for x in hashable["c"])


def test_compute_schema():
    df = pd.DataFrame({"one": [1, 2], "two": ["a", "b"]})
    schema = DatasetLoader.compute_schema(df)
    assert set(schema.columns) == {"col", "dtype"}
    assert "one" in schema["col"].values


def test_preprocess_list_strings_column_basic():
    loader = DatasetLoader()
    # ensure issues dict has expected structure used by the helper
    loader.issues = {"nan": {}}
    s = pd.Series(["['Apple', 'Banana']", "['pear']", None], name="ingredients")
    out = loader._preprocess_list_strings_column(s)
    # should return lists (not strings) and lowercase
    assert isinstance(out.iloc[0], list)
    assert out.iloc[0][0] == "apple"
    assert out.iloc[1] == ["pear"]
    # None should be converted to empty list
    assert out.iloc[2] == []
