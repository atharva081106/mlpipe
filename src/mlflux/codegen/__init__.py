"""
Code generation package for MLFlux.
Generates self-contained, standalone Python scripts and Jupyter Notebooks reproducing fitted pipelines.
"""

from mlflux.codegen.generator import generate_standalone_code, generate_standalone_notebook

__all__ = ["generate_standalone_code", "generate_standalone_notebook"]
