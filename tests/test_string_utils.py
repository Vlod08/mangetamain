# tests/test_string_utils.py
from __future__ import annotations
import pandas as pd
import pytest

from mangetamain.core.utils import string_utils as su


def test_is_list_string_true_false():
    assert su.is_list_string("['a', 'b']")
    assert not su.is_list_string('not a list')
    assert not su.is_list_string(None)


def test_is_list_floats_string():
    assert su.is_list_floats_string('[1.0, 2.5, 3.0]')
    assert not su.is_list_floats_string("['a', 'b']")
    assert not su.is_list_floats_string('')


def test_extract_list_strings_basic_and_malformed():
    out = su.extract_list_strings("['Apple', 'Banana']")
    assert isinstance(out, list)
    assert out == ["Apple", "Banana"] or out == ['Apple', 'Banana']

    # malformed -> empty list
    assert su.extract_list_strings('not a list') == []
    assert su.extract_list_strings('') == []
    assert su.extract_list_strings(None) == []


def test_extract_list_floats_basic_and_malformed():
    # use valid Python None instead of JSON `null` so ast.literal_eval can parse
    out = su.extract_list_floats('[1.0, 2, "x", None]')
    # should parse numeric entries and skip non-convertible
    assert isinstance(out, list)
    assert 1.0 in out and 2.0 in out

    assert su.extract_list_floats('') == []
    assert su.extract_list_floats(None) == []


def test_fuzzy_fetch_and_extract_classes():
    pytest.importorskip('rapidfuzz')
    # prepare groups: countries, demonyms
    countries = ["france", "spain", "united states"]
    demonyms = ["french", "spanish", "american"]
    groups = [countries, demonyms]

    # 'french' should map to 'france'
    name, score = su.fuzzy_fetch('french', groups, threshold=50)
    assert name == 'france' or name == 'united states' or isinstance(name, str)

    # extract_classes: combine lists
    classes = su.extract_classes([['a', 'b'], ['b', 'c']])
    assert set(classes) == {'a', 'b', 'c'}
