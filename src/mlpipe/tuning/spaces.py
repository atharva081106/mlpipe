"""Tuning budgets and search configuration."""


def get_search_iterations(mode: str) -> int:
    """Return number of RandomizedSearchCV iterations for the training mode."""
    mode = str(mode).lower()
    if mode == "fast":
        return 3
    elif mode == "thorough":
        return 20
    return 8  # balanced
