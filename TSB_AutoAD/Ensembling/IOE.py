#  This function is adapted from [UOMS] by [yzhao062]
#  Original source: [https://github.com/yzhao062/UOMS]

import numpy as np
from scipy.spatial import distance
from scipy.stats import rankdata, kendalltau

def Unsupervised_ens(det_scores):

    rank_mat = rankdata(det_scores, axis=0)
    inv_rank_mat = 1 / rank_mat
    n_models = det_scores.shape[1]

    # build target vector 
    target = np.mean(inv_rank_mat, axis=1)
    kendall_vec = np.full([n_models,], -99).astype(float)
    kendall_tracker = []
    
    model_ind = list(range(n_models))
    selected_ind = []
    last_kendall = 0
    
    # build the first target
    for i in model_ind:
        kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]
    most_sim_model = np.argmax(kendall_vec)
    kendall_tracker.append(np.max(kendall_vec))
    
    # option 1: last one: keep increasing/non-decreasing
    # last_kendall = kendall_tracker[-1]
    # option 2: moving avg
    # last_kendall = np.mean(kendall_tracker[-1*moving_size:])
    # option 3: average of all
    last_kendall = np.mean(kendall_tracker)
    
    selected_ind.append(most_sim_model)
    model_ind.remove(most_sim_model)
    
    while len(model_ind) != 0:
        target = np.mean(inv_rank_mat[:, selected_ind], axis=1)
        kendall_vec = np.full([n_models,], -99).astype(float)
        
        for i in model_ind:
            kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]            
        most_sim_model = np.argmax(kendall_vec)
        max_kendall = np.max(kendall_vec)
        
        if max_kendall >= last_kendall:
            selected_ind.append(most_sim_model)
            model_ind.remove(most_sim_model)
            kendall_tracker.append(max_kendall)
            
            # option 1: last one: keep increasing/non-decreasing
            # last_kendall = kendall_tracker[-1]                 
            # # option 2: moving avg
            # last_kendall = np.mean(kendall_tracker[-1*moving_size:])                 
            # option 3: average of all
            last_kendall = np.mean(kendall_tracker)
        else:
            break

    # print('kendall_tracker: ', kendall_tracker)
    # print('selected_ind: ', selected_ind)
    final_target = np.mean(det_scores[:, selected_ind], axis=1)
    return final_target