#  This function is adapted from [tsad-model-selection] by [mononitogoswami]
#  Original source: [https://github.com/mononitogoswami/tsad-model-selection]

from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
import numpy as np
from ..utils import downsample_ts

def Model_Centrality(scores, n_neighbors):
            
    score_ds = downsample_ts(scores, rate=10)
    score_ds = score_ds.T
    ranked_scores = rankdata(score_ds, axis=1)

    MC_list = []
    if isinstance(n_neighbors, int):
        n_neighbors = [n_neighbors]

    neigh = NearestNeighbors(n_neighbors=np.max(n_neighbors),
                             algorithm='ball_tree',
                             metric=kendalltau_dist,
                             n_jobs=4)
    neigh.fit(ranked_scores)

    for nn in n_neighbors:
        MC_list.append(neigh.kneighbors(ranked_scores, n_neighbors=nn)[0].mean(axis=1))
    if not isinstance(n_neighbors, int):
        MC_list = np.array(MC_list).mean(axis=0).tolist()
    return MC_list


def kendalltau_dist(A, B=None):
    """
    This function computes Kendall's-tau distance between two permutations
    using Merge sort algorithm.
    If only one permutation is given, the distance will be computed with the
    identity permutation as the second permutation
   Parameters
   ----------
   A: ndarray
        The first permutation
   B: ndarray, optional
        The second permutation (default is None)
   Returns
   -------
   int
        Kendall's-tau distance between both permutations (equal to the number of inversions in their composition).
    """
    if B is None: B = list(range(len(A)))

    A = np.asarray(A).copy()
    B = np.asarray(B).copy()
    n = len(A)

    # check if A contains NaNs
    msk = np.isnan(A)
    indexes = np.array(range(n))[msk]

    if indexes.size:
        A[indexes] = n  #np.nanmax(A)+1

    # check if B contains NaNs
    msk = np.isnan(B)
    indexes = np.array(range(n))[msk]

    if indexes.size:
        B[indexes] = n  #np.nanmax(B)+1

    inverse = np.argsort(B)
    compose = A[inverse]
    _, distance = mergeSort_rec(compose)

    return distance

def mergeSort_rec(lst):
    """
    This function splits recursively lst into sublists until sublist size is 1. Then, it calls the function merge()
    to merge all those sublists to a sorted list and compute the number of inversions used to get that sorted list.
    Finally, it returns the number of inversions in lst.
    Parameters
    ----------
    lst: ndarray
        The permutation
    Returns
    -------
    result: ndarray
        The sorted permutation
    d: int
        The number of inversions.
    """
    lst = list(lst)
    if len(lst) <= 1:
        return lst, 0
    middle = int(len(lst) / 2)
    left, a = mergeSort_rec(lst[:middle])
    right, b = mergeSort_rec(lst[middle:])
    sorted_, c = merge(left, right)
    d = (a + b + c)
    return sorted_, d

def merge(left, right):
    """
    This function uses Merge sort algorithm to count the number of
    inversions in a permutation of two parts (left, right).
    Parameters
    ----------
    left: ndarray
        The first part of the permutation
    right: ndarray
        The second part of the permutation
    Returns
    -------
    result: ndarray
        The sorted permutation of the two parts
    count: int
        The number of inversions in these two parts.
    """
    result = []
    count = 0
    i, j = 0, 0
    left_len = len(left)
    while i < left_len and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += left_len - i
            j += 1
    result += left[i:]
    result += right[j:]

    return result, count