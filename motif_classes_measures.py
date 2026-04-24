"""
THM classes.
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon

__all__ = [
    "build_dataframe",
    "compute_proportions",
    "compute_absolute_densities",
]


#-=======================================================

classes = ["persistent", "contraction", "expansion", "reconfiguration"]

def build_dataframe(data):
    rows = []
    for subj, periods in data.items():
        for period, values in periods.items():

            counts = {c: values[c] for c in classes}
            total_observed = sum(counts.values())
            total_possible = values["possible_motifs"]

            proportions = {f"{c}_prop": counts[c] / total_observed for c in classes}
            densities = {f"{c}_dens": counts[c] / total_possible for c in classes}

            row = {
                "subject": subj,
                "period": period,
                **counts,
                **proportions,
                **densities
            }
            rows.append(row)
    return pd.DataFrame(rows)


def compute_proportions(data):
    """Proportions.
    """
    rows = []
    for subj, periods in data.items():
        for period, counts in periods.items():
            total = sum(counts[c] for c in classes)

            proportions = {c: counts[c] / total for c in classes}
            row = {
                "subject": subj,
                "period": period,
                **proportions
            }
            rows.append(row)
    return pd.DataFrame(rows)


def compute_absolute_densities(data):
    """Absolute densities.
    """
    rows = []
    for subj, periods in data.items():
        for period, info in periods.items():
            counts = info["counts"]
            total_possible = info["possible_motifs"]

            densities = {
                c: counts[c] / total_possible for c in classes
            }

            row = {
                "subject": subj,
                "period": period,
                **densities
            }
            rows.append(row)
    return pd.DataFrame(rows)
