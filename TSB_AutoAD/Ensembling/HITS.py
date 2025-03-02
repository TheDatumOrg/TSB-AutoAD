#  This function is adapted from [UOMS] by [yzhao062]
#  Original source: [https://github.com/yzhao062/UOMS]

import numpy as np
from scipy.stats import rankdata

def HITS(det_scores):

    rank_mat = rankdata(det_scores, axis=0)
    # inv_rank_mat = 1 / rank_mat
    inv_rank_mat = rank_mat
    n_samples = det_scores.shape[0]
    n_models = det_scores.shape[1]
    
    hub_vec = np.full([n_models, 1],  1/n_models)
    auth_vec = np.zeros([n_samples, 1])
    hub_vec_list = []
    auth_vec_list = []
    
    hub_vec_list.append(hub_vec)
    auth_vec_list.append(auth_vec)

    max_iter = 500

    for i in range(max_iter):
        auth_vec = np.dot(inv_rank_mat, hub_vec)
        auth_vec = auth_vec/np.linalg.norm(auth_vec)
        
        # update hub_vec
        hub_vec = np.dot(inv_rank_mat.T, auth_vec)
        hub_vec = hub_vec/np.linalg.norm(hub_vec)
        
        # stopping criteria
        auth_diff = auth_vec - auth_vec_list[-1]
        hub_diff = hub_vec - hub_vec_list[-1]
        
        # print(auth_diff.sum(), auth_diff.mean(), auth_diff.std())
        # print(hub_diff.sum(), hub_diff.mean(), hub_diff.std())
        # print()

        if np.abs(auth_diff.sum()) <= 1e-10 and np.abs(auth_diff.mean()) <= 1e-10 and np.abs(hub_diff.sum()) <= 1e-10 and np.abs(hub_diff.mean()) <= 1e-10:
            # print('Iterative_mc break at', i)
            break
        
        auth_vec_list.append(auth_vec)
        hub_vec_list.append(hub_vec)

    return auth_vec.ravel()
