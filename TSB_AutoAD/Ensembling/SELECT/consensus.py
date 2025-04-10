from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .kemeny_young import kemeny_young_iterative
from .rra import robust_rank_aggregation_impl
from .gaussian_scaler import GaussianScaler
from .jings_method import unify_em_jings

def scores_to_ranks(scores: np.ndarray, invert_order: bool = False) -> np.ndarray:
    """Converts scores to ranks.

    If the input consists of anomaly scores, the ranks should be inverted so that higher scores get lower ranks. Use
    ``invert_order=True`` for that. E.g.:

    >>> scores = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.1]]).T
    >>> scores_to_ranks(scores, invert_order=True)
    array([[3, 2, 1],
           [2, 1, 3]])

    Parameters
    ----------
    scores : np.ndarray
        The scores to convert to ranks. Each row represents the scores for one instance. If a 1-dim. array is passed,
        the ranks are computed for that array.
    invert_order : bool
        Whether to invert the order of the ranks so that highest score gets rank 1. Defaults to False so that lowest
        score gets rank 1.

    Returns
    -------
    np.ndarray
        The ranks for each score. Each row represents the ranks for one instance.
    """
    ranks = rankdata(scores, method="max", axis=1 if scores.ndim > 1 else None)
    ranks = ranks.astype(np.int_)
    if invert_order:
        ranks = ranks.max() - ranks + 1
    return ranks

def ranks_to_scores(ranks: np.ndarray) -> np.ndarray:
    """Converts ranks to scores.

    Parameters
    ----------
    ranks : np.ndarray
        The ranks to convert to scores. Each row represents the ranks for one instance. If a 1-dim. array is passed,
        the scores are computed for that array.

    Returns
    -------
    np.ndarray
        The scores for each rank. Each row represents the scores for one instance.
    """
    scores = 1. / ranks
    # scores = ranks.max() - ranks
    # scores = scores.astype(np.float_)
    # if scores.ndim == 1:
    #     scores = scores.reshape(-1, 1)
    # scores = MinMaxScaler().fit_transform(scores)
    # if scores.ndim == 1:
    #     scores = scores.ravel()
    return scores

def inverse_ranking(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i_ranks = 1 / ranks
    i_ranks = np.mean(i_ranks, axis=0)
    scores = i_ranks.copy()
    ranks = rankdata(1 - i_ranks, method="max")
    return ranks, scores


def kemeny_young(ranks: np.ndarray, approx: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    ranks, scores = kemeny_young_iterative(ranks.T, approx=approx)
    return ranks, scores


def robust_rank_aggregation(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return robust_rank_aggregation_impl(ranks)


def mixture_modeling(scores: np.ndarray, aggregation, cut_off: int = 20, iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    scores = unify_em_jings(scores, cut_off, iterations, n_jobs=1)
    scores = aggregation(scores, axis=0)
    ranks = scores_to_ranks(1 - scores)
    return ranks, scores


def unification(scores: np.ndarray, aggregation) -> Tuple[np.ndarray, np.ndarray]:
    scores = GaussianScaler().fit_transform(scores)
    scores = aggregation(scores, axis=0)
    ranks = scores_to_ranks(1 - scores)
    return ranks, scores


if __name__ == '__main__':
    # ranks = np.array([[3, 1, 4, 2, 2], [1, 2, 3, 4, 3], [1, 2, 4, 3, 5]])
    debug_names = np.array(["Memphis", "Nashville", "Knoxville", "Chattanooga"])
    ranks = np.vstack([
        np.tile([1, 2, 4, 3], (42, 1)),  # Memphis
        np.tile([4, 1, 3, 2], (26, 1)),  # Nashville
        np.tile([4, 3, 2, 1], (15, 1)),  # Chattanooga
        np.tile([4, 3, 1, 2], (17, 1)),  # Knoxville
    ])
    scores = ranks_to_scores(ranks)
    print("Ranks:", ranks.shape)
    print("Ranks:", ranks)

    def _get_ranking(ranks: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({"rank": ranks, "name": debug_names})
        return df.sort_values(by="rank")["name"].values

    # final_ranks, final_scores = inverse_ranking(ranks)
    final_ranks, final_scores = kemeny_young(ranks, approx=False)
    # final_ranks, final_scores, _, _ = robust_rank_aggregation(ranks)
    # final_ranks, final_scores = mixture_modeling(scores, aggregation=np.max)
    # final_ranks, final_scores = unification(scores, aggregation=np.mean)
    print("FinalRank:", final_ranks)
    print("Scores:", final_scores)

    print("                   ", debug_names)
    print("First ranking:     ", ranks[0, :])
    print("Aggregated:        ", final_ranks)
    print("First ranking:", _get_ranking(ranks[0, :]))
    print("Aggregated ranking:", _get_ranking(final_ranks))
