"""Tests for model candidates and registry."""

from mlpipe.models.classification import get_classification_candidates
from mlpipe.models.regression import get_regression_candidates
from mlpipe.models.selection import get_candidates_for_task


def test_classification_candidates():
    candidates = get_classification_candidates()
    names = [c.name for c in candidates]
    assert "Logistic Regression" in names
    assert "Random Forest" in names
    assert "HistGradientBoosting" in names
    assert "Decision Tree" in names
    assert "K-Nearest Neighbors" in names

    for c in candidates:
        est = c.create_estimator(random_seed=42)
        assert est is not None
        assert len(c.get_search_space("fast")) >= 0
        assert len(c.get_search_space("balanced")) >= 0


def test_regression_candidates():
    candidates = get_regression_candidates()
    names = [c.name for c in candidates]
    assert "Ridge" in names
    assert "Random Forest Regressor" in names
    assert "HistGradientBoosting Regressor" in names
    assert "Decision Tree Regressor" in names

    for c in candidates:
        est = c.create_estimator(random_seed=42)
        assert est is not None


def test_get_candidates_for_task():
    clf_cands = get_candidates_for_task("classification")
    assert len(clf_cands) >= 4
    reg_cands = get_candidates_for_task("regression")
    assert len(reg_cands) >= 4
