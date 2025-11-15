"""
Utility package for working with FCC interference datasets.

Provides helpers for loading TV graphs, generating subgraphs, and summarising
constraint statistics.
"""

from .tv_graph import Interference, Station, TVGraph

__all__ = ["Interference", "Station", "TVGraph"]

