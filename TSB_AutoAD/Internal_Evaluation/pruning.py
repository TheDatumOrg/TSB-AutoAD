from __future__ import annotations

import logging
from itertools import combinations
from typing import Tuple, Set

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from dataset import TrainingDatasetCollection, TrainDataset


def filter_similar_timeseries(train_collection):
    """Dataset similarity pruning:

    Check if two timeseries are similar (i.e. their distance is below a threshold), then remove the one whose period
    size is less frequent in the dataset.
    """

    length_groups = {}
    period_size_groups = {}
    for d in train_collection.datasets:
        length_groups.setdefault(d.length, []).append(d)
        period_size_groups.setdefault(d.period_size, []).append(d)

    for length, group in length_groups.items():
        data = np.vstack([d.data for d in group])
        data = MinMaxScaler().fit_transform(data.T).T  # FIXME: shouldn't we rather use StandardScaler?
        dists = euclidean_distances(data)
        threshold = 0.0005 * length
        dists = dists < threshold

        to_remove: Set[TrainDataset] = set()
        i = 0
        while i < dists.shape[0]:
            sim_idxs = np.arange(dists.shape[0])[dists[i, :]]
            if len(sim_idxs) > 1:
                ps_cardinalities = np.array([len(period_size_groups[group[i].period_size]) for i in sim_idxs])
                idx_to_remove = sim_idxs[np.argsort(ps_cardinalities)[-(len(sim_idxs) - 1):]]

                for idx in idx_to_remove:
                    d = group[idx]
                    if d in period_size_groups[d.period_size]:
                        period_size_groups[d.period_size].remove(d)
                    to_remove.add(d.name)
                remaining_dists_idxs = [i for i in np.arange(dists.shape[0]) if i not in idx_to_remove]
                dists = dists[remaining_dists_idxs, :][:, remaining_dists_idxs]
            i += 1
        # actually remove the datasets from the collection
        train_collection.datasets = [d for d in train_collection.datasets if d.name not in to_remove]
