from __future__ import annotations

import logging
import multiprocessing
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def slice_lengths(slices: np.ndarray, absolute: bool = False) -> np.ndarray:
    """Compute the (absolute) length of each slice (start, end) by computing length = end - start. If ``absolute`` is
    True, the absolute length is computed.

    Parameters
    ----------
    slices : np.ndarray
        A 2D integer array containing a vector of (start, end)-slices.
    absolute : bool, optional
        Whether to return the absolute length (default: False).

    Returns
    -------
    lengths : np.ndarray
        A 1D integer array containing the length of each slice (end - start).
    """
    length = np.diff(slices, axis=1).ravel()
    if absolute:
        return np.abs(length, dtype=np.int_)
    else:
        return length.astype(np.int_)

class TimeseriesLimiter:
    def __init__(self):
        self.a = 0

    def _find_subsequence_with_least_cuts(self, dataset, desired_length):
        self._timers.start(f"Find least cuts-{dataset.name}")
        n_cuts = len(dataset.cut_points())
        cuts = np.r_[0, dataset.cut_points(), dataset.length].astype(np.int_)
        cut_indices = np.arange(0, n_cuts + 2)
        # print(f"{cuts=}")
        for allowed_cuts in range(n_cuts + 1):
            space = np.roll(cuts, -(allowed_cuts + 1)) - cuts
            # print(f"{space=}")
            idx = cut_indices[space >= desired_length]
            if len(idx) > 0:
                middle = int(np.ceil((len(idx) - 1) / 2))
                subsequence_idx = cuts[idx[middle]]
                self._log.debug(f"... found idx={subsequence_idx} with {allowed_cuts} cuts.")
                self._timers.stop(f"Find least cuts-{dataset.name}")
                return subsequence_idx

    def _get_length_limit(self, period_size, soft=False) -> int:
        length_limit = max(
            2000,
            period_size * 10)
        # length_limit = 1000
        if soft:
            length_limit = int(length_limit * 1.5)
        return length_limit

    def extract_sampled_regime(self, collection,
                               period_size,
                               in_place = True):
        """Implements simple random sampling as base behavior extraction strategy.

        If the dataset is smaller than the desired length, the whole dataset is used.
        If we can only extract all desired samples by sampling uniformly from the whole dataset, we do so.
        """
        max_samples = 2
        max_overlap = 0.25
        rng = np.random.default_rng(self._general_config.seed)

        dataset = collection.test_data
        desired_length = self._get_length_limit(period_size, soft=True)
        if max_samples < 1:
            max_samples = 1

        if dataset.length < desired_length:
            # take everything
            masks = np.ones((1, dataset.length), dtype=np.bool_)

        else:
            n_samples = min(
                max_samples,
                int((dataset.length - desired_length * max_overlap) / ((1 - max_overlap) * desired_length))
            )
            if n_samples == 1:
                # sample a single random subsequence
                random_idx = rng.integers(low=0, high=dataset.length - desired_length, endpoint=True)
                start_idxs = np.array([random_idx], dtype=np.int_)
            elif (n_samples + 1) * desired_length - n_samples * (desired_length * max_overlap) > dataset.length:
                # sample uniformly
                if n_samples == 2:
                    # sample two evenly spaced subsequences
                    spacing = dataset.length // 4
                    start_idxs = np.array([spacing - desired_length//2, 3*spacing - desired_length//2], dtype=np.int_)
                else:
                    # evenly space subsequences from both ends
                    start_idxs = np.linspace(0, dataset.length - desired_length, n_samples, endpoint=True, dtype=np.int_)
            else:
                # sample randomly (greedy)
                start_idxs = np.zeros(n_samples, dtype=np.int_)
                i = 0
                n_tries = 0
                while i < n_samples and n_tries < 500:
                    n_tries += 1
                    idx = rng.integers(low=0, high=dataset.length - desired_length, endpoint=True)
                    if not (
                            np.any(  # check if begin is in any of the already selected slices
                                (start_idxs[:i] < idx) &
                                (idx <= start_idxs[:i] + desired_length * (1 - max_overlap))
                            ) or
                            np.any(  # check if end is in any of the already selected slices
                                (start_idxs[:i] + desired_length * max_overlap <= idx + desired_length) &
                                (idx + desired_length < start_idxs[:i] + desired_length)
                            )
                    ):
                        start_idxs[i] = idx
                        i += 1

                if i < n_samples:
                    n_samples = i
                    start_idxs = start_idxs[:i]

            masks = np.zeros((n_samples, dataset.length), dtype=np.bool_)
            for i, idx in enumerate(start_idxs):
                masks[i, idx:idx + desired_length] = True

        if in_place:
            for i in range(masks.shape[0]):
                collection.add_base_ts(mask=masks[i, :], period_size=period_size)
        else:
            return masks

    def _select_slices(self, slices: np.ndarray, desired_length: int) -> np.ndarray:
        # sort slices according to their length and add them incrementally until desired length is exceeded
        sls = slice_lengths(slices)
        sorted_idxs = np.argsort(sls)[::-1]
        # sum up lengths until desired length is exceeded (+ next slice because we want to be larger)
        indices = np.nonzero(np.cumsum(sls[sorted_idxs]) < desired_length)[0]
        if indices.shape[0] == 0:  # if first slice is already larger than desired length, use it
            indices = [0]
        elif indices.shape[0] == sorted_idxs.shape[0]:  # if we need all slices, use all
            pass
        else:  # if we selected a specific number of slices, we need to add one to exceed the desired length
            indices = np.r_[indices, indices[-1] + 1]
        # print(f"{sorted_idxs=}")
        # print(f"{indices=}")
        indices = sorted_idxs[indices]
        self._log.debug(f"Removing {slices.shape[0] - indices.shape[0]} excess small slices; "
                        f"remaining length={int(np.cumsum(sls[indices])[-1])}.")
        slices = slices[indices]
        # cap the last slice to the desired length
        # FIXME: guard against too small slices
        length_reduction = int(np.cumsum(sls[indices])[-1] - desired_length)
        slices[-1, 1] -= length_reduction
        self._log.debug(f"Reducing last slice by {length_reduction} to reach desired length={desired_length}")
        sls = slice_lengths(slices)
        self._log.debug(f"Selected slices={slices.tolist()} and their lengths={sls.tolist()}")
        return slices


def limit_base_timeseries(collection):

    limiter = TimeseriesLimiter()
    for d in collection:
        limiter.enforce_length_limit(d)
    return collection
