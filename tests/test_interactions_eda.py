# tests/test_interactions_eda.py
from __future__ import annotations

def test_rating_range(interactions_service):
    lo, hi = interactions_service.rating_range()
    assert (lo, hi) == (2.0, 5.0)

def test_text_features_exist(interactions_service):
    df = interactions_service.with_text_features()
    expected = {
        "review_len",
        "review_words",
        "review_exclamations",
        "review_question_marks",
        "review_has_caps",
        "review_mentions_thanks",
    }
    assert expected.issubset(df.columns)

def test_histograms(interactions_service):
    h1 = interactions_service.hist_rating()
    h2 = interactions_service.hist_review_len()
    assert {"left","right","count"}.issubset(h1.columns)
    assert {"left","right","count"}.issubset(h2.columns)

def test_user_bias(interactions_service):
    ub = interactions_service.user_bias()
    assert {"user_id","n","mean","median"}.issubset(ub.columns)
    assert int(ub[ub["user_id"] == 3]["n"].iloc[0]) == 3

def test_tokens_by_rating(interactions_service):
    tbr = interactions_service.tokens_by_rating(k=5)
    assert {"rating","token","count"}.issubset(tbr.columns)
    assert (tbr["token"].astype(str).str.len() > 0).all()

def test_time_series_bundle(interactions_service):
    bm = interactions_service.by_month()
    assert {"month","n"}.issubset(bm.columns)
    roll = interactions_service.monthly_rolling(window=2)
    assert any(c.startswith("n_roll") for c in roll.columns)
    yoy = interactions_service.monthly_yoy()
    if not yoy.empty:
        assert "n_yoy" in yoy.columns

def test_weekday_heat_and_cohorts(interactions_service):
    wk = interactions_service.weekday_profile()
    assert {"wk","n"}.issubset(wk.columns)
    mat = interactions_service.weekday_hour_heat()
    assert {"wk","h","n"}.issubset(mat.columns)
    coh = interactions_service.cohorts_users()
    if not coh.empty:
        assert {"cohort","age","n"}.issubset(coh.columns)
