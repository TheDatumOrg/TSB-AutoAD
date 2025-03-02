#  This function is adapted from [EMMV_benchmarks] by [ngoix]
#  Original source: [https://github.com/ngoix/EMMV_benchmarks]

import numpy as np
from sklearn.metrics import auc
from sklearn.utils import shuffle as sh
import os
from sklearn.preprocessing import MinMaxScaler

def em(t, t_max, volume_support, s_unif, s_X, n_generated):
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.
    for u in s_X_unique:
        # if (s_unif >= u).sum() > n_generated / 1000:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                          t * (s_unif > u).sum() / n_generated
                          * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        print ('\n failed to achieve t_max \n')
        amax = -1
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


def mv(axis_alpha, volume_support, s_unif, s_X, n_generated):
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        # pdb.set_trace()
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return auc(axis_alpha, mv), mv

def Excess_Mass(args, data, Candidate_Model_Set, alpha_min=0.9, alpha_max=0.999,
                       n_generated=10000, t_max = 0.9):
    
    em_list = []
    data = data.reshape(-1, 1)
    n_features = data.shape[1]
    
    lim_inf = data.min(axis=0)
    lim_sup = data.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    if volume_support == 0:
        volume_support = ((lim_sup - lim_inf) + 0.001).prod()
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features))

    for i, det in enumerate(Candidate_Model_Set):

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

        path_unif = f'{args.score_unif_dir}/{det}/{args.filename.split(".")[0]}.npy'
        if os.path.exists(path_unif):
            score_unif = np.load(path_unif)
        else:
            print('No score_unif found, use random score instead')
            anomaly_score_pool = []
            for i in range(5):
                anomaly_score_pool.append(np.random.uniform(size=args.ts_len))
            score_unif = np.mean(np.array(anomaly_score_pool), axis=0)

        if len(score_unif) < args.ts_len:
            score = np.pad(score_unif, (0, args.ts_len - len(score_unif)), mode='constant')
        elif len(score_unif) > args.ts_len:
            score_unif = score_unif[:args.ts_len]

        score_unif = MinMaxScaler(feature_range=(0,1)).fit_transform(score_unif.reshape(-1,1)).ravel()


        s_X_clf = score* -1         # reverse the score
        s_unif_clf = score_unif* -1
        em_clf = em(t, t_max, volume_support, s_unif_clf, s_X_clf, n_generated)[0]* -1
        em_list.append(em_clf)
        # print('det: {}, em_clf: {}'.format(det, em_clf))
    
    return em_list

def Mass_Volume(args, data, Candidate_Model_Set, alpha_min=0.9, alpha_max=0.999,
                       n_generated=10000, t_max = 0.9):
    
    mv_list = []
    data = data.reshape(-1, 1)
    n_features = data.shape[1]
    
    lim_inf = data.min(axis=0)
    lim_sup = data.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    if volume_support == 0:
        volume_support = ((lim_sup - lim_inf) + 0.001).prod()
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features))

    for i, det in enumerate(Candidate_Model_Set):

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

        path_unif = f'{args.score_unif_dir}/{det}/{args.filename.split(".")[0]}.npy'
        if os.path.exists(path_unif):
            score_unif = np.load(path_unif)
        else:
            print('No score_unif found, use random score instead')
            anomaly_score_pool = []
            for i in range(5):
                anomaly_score_pool.append(np.random.uniform(size=args.ts_len))
            score_unif = np.mean(np.array(anomaly_score_pool), axis=0)

        if len(score_unif) < args.ts_len:
            score = np.pad(score_unif, (0, args.ts_len - len(score_unif)), mode='constant')
        elif len(score_unif) > args.ts_len:
            score_unif = score_unif[:args.ts_len]

        score_unif = MinMaxScaler(feature_range=(0,1)).fit_transform(score_unif.reshape(-1,1)).ravel()


        s_X_clf = score* -1         # reverse the score
        s_unif_clf = score_unif* -1
        mv_clf = mv(axis_alpha, volume_support, s_unif_clf, s_X_clf, n_generated)[0]
        mv_list.append(mv_clf)
        # print('det: {}, mv_clf: {}'.format(det, mv_clf))
    
    return mv_list   # EM * -1, MV smaller better

unsupervised_evaluation_curves = {
    "EM": Excess_Mass,
    "MV": Mass_Volume
}

def run_unsupervised_evaluation_curves(variant, data, Candidate_Model_Set, args):
    autoad = unsupervised_evaluation_curves.get(variant)
    if autoad:
        return autoad(args, data, Candidate_Model_Set)
    else:
        raise NotImplementedError