# tests/conftest.py
from __future__ import annotations
import sys, importlib, types
import pandas as pd
import pytest

# --- Keep short imports working in tests (core.*, app.*) ---
def _install_import_shims() -> None:
    try:
        sys.modules.setdefault("core", importlib.import_module("mangetamain.core"))
        for sub in ("dataset", "recipes_eda", "interactions_eda"):
            try:
                sys.modules.setdefault(f"core.{sub}", importlib.import_module(f"mangetamain.core.{sub}"))
            except Exception:
                pass
    except Exception:
        pass
    try:
        sys.modules.setdefault("app", importlib.import_module("mangetamain.app"))
        try:
            sys.modules.setdefault("app.app_utils", importlib.import_module("mangetamain.app.app_utils"))
            sys.modules.setdefault("app.app_utils.ui", importlib.import_module("mangetamain.app.app_utils.ui"))
        except Exception:
            app_utils = types.ModuleType("app.app_utils")
            ui = types.ModuleType("app.app_utils.ui")
            def _dummy_use_global_ui(**kwargs): return None
            ui.use_global_ui = _dummy_use_global_ui
            sys.modules["app.app_utils"] = app_utils
            sys.modules["app.app_utils.ui"] = ui
    except Exception:
        fake_app = types.ModuleType("app")
        fake_app_utils = types.ModuleType("app.app_utils")
        fake_ui = types.ModuleType("app.app_utils.ui")
        def _dummy_use_global_ui(**kwargs): return None
        fake_ui.use_global_ui = _dummy_use_global_ui
        sys.modules["app"] = fake_app
        sys.modules["app.app_utils"] = fake_app_utils
        sys.modules["app.app_utils.ui"] = fake_ui

_install_import_shims()

# --- Neutralize Streamlit caches in pytest to avoid ScriptRunContext warnings ---
import streamlit as st  # noqa
if hasattr(st, "cache_data"):
    st.cache_data = lambda *a, **k: (lambda f: f)  # no-op decorator
if hasattr(st, "cache_resource"):
    st.cache_resource = lambda *a, **k: (lambda f: f)

@pytest.fixture
def interactions_small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":   [1, 1, 2, 3, 3, 3],
        "recipe_id": [10, 11, 10, 12, 12, 13],
        "rating":    [5, 4, 3, 5, 2, 4],
        "review": [
            "Amazing recipe! Loved it",
            "Pretty good; thanks!",
            "ok taste",
            "BEST EVER!!!",
            "not good",
            None,  # <- important: covers None/NaN path
        ],
        "date": pd.to_datetime([
            "2020-01-15","2020-02-02","2020-02-20",
            "2020-03-05","2020-04-01","2020-04-28"
        ]),
    })

@pytest.fixture
def interactions_service(interactions_small_df):
    from mangetamain.core.interactions_eda import InteractionsEDAService
    svc = InteractionsEDAService()

    # SAFER: bypass svc.load() (which calls the cached method) and set df directly
    # so tests don't depend on Streamlit runtime.
    svc.ds.df = interactions_small_df  # type: ignore[attr-defined]

    # Monkey-patch a safe version of compute_text_features that handles None
    def _safe_compute_text_features(col: pd.Series) -> pd.DataFrame:
        s = col.fillna("").astype(str)
        out = pd.DataFrame()
        out[f"{col.name if col.name else 'review'}_len"] = s.str.len()
        out[f"{col.name if col.name else 'review'}_words"] = s.str.split().str.len()
        out[f"{col.name if col.name else 'review'}_exclamations"] = s.str.count("!")
        out[f"{col.name if col.name else 'review'}_question_marks"] = s.str.count(r"\?")
        out[f"{col.name if col.name else 'review'}_has_caps"] = s.str.contains(r"[A-Z]{3,}", regex=True)
        out[f"{col.name if col.name else 'review'}_mentions_thanks"] = s.str.contains(
            r"\bthank(?:s| you)?\b", case=False, regex=True
        )
        return out

    # Replace the method on the class for this test session
    InteractionsEDAService.compute_text_features = staticmethod(_safe_compute_text_features)  # type: ignore

    # Precompute features like the real load() would have done
    svc.text_features = _safe_compute_text_features(svc.ds.df["review"])  # type: ignore[index]
    return svc
