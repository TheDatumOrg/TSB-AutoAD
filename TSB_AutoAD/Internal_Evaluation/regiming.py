from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import stumpy
from matplotlib import pyplot as plt

from .limiting import TimeseriesLimiter
from .dataset import TrainingDatasetCollection

tiny = np.finfo(np.float_).tiny

def mask_to_slices(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean vector mask to index slices (inclusive start, exclusive end) where the mask is ``True``.

    Parameters
    ----------
    mask : numpy.ndarray
        A boolean 1D array

    Returns
    -------
    slices : numpy.ndarray
        (-1, 2)-shaped array of slices. Each slice consists of the start index (inclusive) and the end index (exclusive)
        of a continuous region of ``True``-values.
    """
    tmp = np.r_[0, mask, 0]
    slices = np.c_[
        np.nonzero(np.diff(tmp) == 1)[0],
        np.nonzero(np.diff(tmp) == -1)[0]
    ]
    return slices

def invert_slices(slices: np.ndarray, first_last: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Inverts the slices, such that the new first slice describes the region between the end of the first slices and
    the beginning of the second slice, and so forth. If ``first_last`` is provided, the first and last points
    are used to construct the first and last slices, respectively.

    Parameters
    ----------
    slices : np.ndarray
        A 2D integer array containing a vector of (start, end)-slices.
    first_last : Tuple[int, int], optional
        The first and last (exclusive) index within the original array.

    Returns
    -------
    inv_slices : np.ndarray
        A 2D integer array containing a vector of (end_i, start_i-1)-slices.
    """
    if first_last is not None:
        first, last = first_last
        slices = np.r_[slices, [[last, first]]]
    ends = slices[:, 1]
    starts = np.roll(slices[:, 0], -1)
    inv_slices = np.c_[ends, starts]
    return inv_slices

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

def find_best_k(areas):
    k_range = np.arange(1, 7 + 1)
    if areas.min() < tiny:
        best_k = k_range[np.argmin(areas)]
    else:
        df_aud = pd.DataFrame({
            "area": areas,
            "area_change": np.r_[np.nan, areas[:-1] / areas[1:] - 1.0],
            "k": k_range
        })
        if df_aud["area_change"].max() < 0.5:
            best_k = 1
        else:
            best_k = df_aud.loc[df_aud["area_change"].argmax(), "k"]

    return best_k


def extract_regimes(profiles, p_idx, period_size):
    regimes = []
    masks = np.zeros_like(profiles, dtype=np.bool_)
    total_min = np.min(profiles[p_idx], axis=0)
    # add a small epsilon to prevent quickly shifting between multiple good matching snippets
    masks[p_idx, :] = profiles[p_idx, :] <= total_min + 1e-6
    slices = {}
    for i in p_idx:
        slices[i] = mask_to_slices(masks[i])

    # add small skipped subsequences to the same regime
    for i in p_idx:
        candidate_slices = slices[i]
        # only consider strong (large enough slices) to compute skipped regions
        strong_slices = candidate_slices[
            slice_lengths(candidate_slices) > 0.8 * period_size
        ]

        inv_slices = invert_slices(strong_slices)

        fix_slices = inv_slices[
            slice_lengths(inv_slices, absolute=True) < 0.95 * period_size
        ]

        for s, e in fix_slices:
            masks[i, s:e] = True

    # limiter = TimeseriesLengthLimiter()
    for i in p_idx:
        # recompute slices
        slices = mask_to_slices(masks[i])

        # only consider strong slices
        mask = slice_lengths(slices) > 0.8 * period_size
        if mask.sum() > 0:
            slices = slices[mask]

        # check if we have many cuts/regime changes (--> small slices), then ignore this snippet and recompute
        median_periods_per_slice = np.median(slice_lengths(slices) // period_size)
        if median_periods_per_slice < 5:
            return extract_regimes(profiles, np.delete(p_idx, np.where(p_idx == i)), period_size)
            # return extract_regimes(np.delete(profiles, i, axis=0), best_k-1, period_size)

        # # select the largest slices until we exceed the desired training dataset length
        # desired_length = limiter._get_length_limit(period_size, soft=True)
        # slices = limiter.select_slices(slices, desired_length=desired_length)

        # adjust slice indices based on sub-window-size
        slices += int(0.5 * period_size)
        regimes.append(np.c_[np.repeat(i, slices.shape[0]), slices])

    return np.vstack(regimes), p_idx


def get_regime_masks(train_collection, period_size):
    if period_size > 600:
        return np.empty((0, train_collection.test_data.length), dtype=np.bool_)
    if period_size < 3 / 0.5:
        return np.empty((0, train_collection.test_data.length), dtype=np.bool_)
    if period_size >= 0.1 * train_collection.test_data.length:
        return np.empty((0, train_collection.test_data.length), dtype=np.bool_)

    # Timers.start(f"Stumpy-{period_size}")
    _snippets, indices, profiles, _fractions, areas, _regimes = stumpy.snippets(
        T=train_collection.test_data.data,
        m=period_size,
        k=7,
        percentage=0.5
    )
    best_k = find_best_k(areas)
    # Timers.stop(f"Stumpy-{period_size}")

    if best_k == 1:
        return TimeseriesLimiter().extract_sampled_regime(train_collection, period_size=period_size, in_place=False)

    # Timers.start(f"Regime extraction-{period_size}")
    # extract snippet regimes
    old_best_k = best_k
    # use only the best k snippets
    p_idx = np.arange(profiles.shape[0])[:best_k]
    regimes, p_idx = extract_regimes(profiles, p_idx, period_size)
    best_k = p_idx.shape[0]

    # create base time series
    masks = np.zeros((best_k, train_collection.test_data.length), dtype=np.bool_)
    for i, snippet_idx in enumerate(p_idx):
        slices_of_indices = regimes[np.where(regimes[:, 0] == snippet_idx)][:, 1:]
        for start_idx, stop_idx in slices_of_indices:
            masks[i, start_idx:stop_idx] = True

    return masks
