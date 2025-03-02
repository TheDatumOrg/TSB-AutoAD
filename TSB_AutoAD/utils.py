import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
import os

def downsample_ts(ts, rate=10):
    ts_len, feat = ts.shape
    right_padding = rate - ts_len % rate if ts_len % rate != 0 else 0
    ts_pad = np.pad(ts, ((0, right_padding), (0, 0)))
    ts_ds = ts_pad.reshape(-1, rate, feat).max(axis=1)
    return ts_ds

def loading_scores(Candidate_Model_Set, args):
    scores_list = []
    for det in Candidate_Model_Set:

        path = f'{args.score_dir}/{det}/{args.filename.split(".")[0]}.npy'
        if os.path.exists(path):
            score = np.load(path)
        else:
            print('No score found, use random score instead')
            anomaly_score_pool = []
            for i in range(5):
                anomaly_score_pool.append(np.random.uniform(size=args.ts_len))
            score = np.mean(np.array(anomaly_score_pool), axis=0)

        if len(score) < args.ts_len:
            score = np.pad(score, (0, args.ts_len - len(score)), mode='constant')
        elif len(score) > args.ts_len:
            score = score[:args.ts_len]

        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        scores_list.append(score)
    det_scores = np.array(scores_list).T  # (score_len, num_score)
    return det_scores

def get_preds_ratio(scores, outliers_ratio = 0.05):
    num = int(len(scores)*outliers_ratio)
    threshold = np.sort(scores)[::-1][num]
    predictions = np.array(scores > threshold)
    predictions = np.array([int(i) for i in predictions])      
    return predictions

def gen_autood_initial_set(data, Candidate_Model_Set, args):

    threshold_ratio_range = [0.02, 0.04, 0.08, 0.10, 0.15, 0.20]

    all_scores = []
    unique_score = []
    all_preds = []

    for det in Candidate_Model_Set:

        path = f'{args.score_dir}/{det}/{args.filename.split(".")[0]}.npy'
        if os.path.exists(path):
            score = np.load(path)
        else:
            print('No score found, use random score instead')
            anomaly_score_pool = []
            for i in range(5):
                anomaly_score_pool.append(np.random.uniform(size=args.ts_len))
            score = np.mean(np.array(anomaly_score_pool), axis=0)

        if len(score) < args.ts_len:
            score = np.pad(score, (0, args.ts_len - len(score)), mode='constant')
        elif len(score) > args.ts_len:
            score = score[:args.ts_len]

        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

        # score = np.load(f'{args.score_dir}/{det}/{args.filename[:-4]}.npy')
        unique_score.append(score)
        for threds in threshold_ratio_range:
            all_preds.append(get_preds_ratio(score, threds))
            all_scores.append(score)

    preds = np.stack(all_preds).T
    scores = np.stack(all_scores).T
    unique_score = np.stack(unique_score).T

    # print('preds shape:', np.shape(preds))
    # print('unique_score shape:', np.shape(unique_score))    # [ts_len, num_det]

    # TODO: Needs to adjust when changing candidate model set
    n = len(Candidate_Model_Set)
    detector_index_ranges = [[i, i + 1] for i in range(n - 1)]
    instance_index_ranges = [[6 * i, 6 * (i + 1)] for i in range(n - 1)]

    return preds, scores, unique_score, instance_index_ranges, detector_index_ranges

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
