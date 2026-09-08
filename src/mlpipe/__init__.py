"""Backward compatibility shim for mlpipe -> mlflux."""
import sys
import mlflux
from mlflux import *  # noqa: F401, F403
from mlflux.core.pipeline import Pipeline  # noqa: F401
from mlflux.core.exceptions import MLFluxError as MLPipeError  # noqa: F401

sys.modules["mlpipe"] = mlflux
