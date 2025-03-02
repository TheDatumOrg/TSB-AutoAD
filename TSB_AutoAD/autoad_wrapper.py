"""
Automated Solution Wrapper
"""

import numpy as np
import pandas as pd
from TSB_AD.models.feature import Window
from TSB_AD.utils.slidingWindows import find_length_rank
import math, os, pickle
from pathlib import Path
import pycatch22 as catch22
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from joblib import load
from sklearn.preprocessing import StandardScaler
import random, glob

from .utils import scores_to_ranks, ranks_to_scores, loading_scores, gen_autood_initial_set

from .Internal_Evaluation.CQ import run_cluster_quality
from .Internal_Evaluation.UEC import run_unsupervised_evaluation_curves
from .Internal_Evaluation.MC import Model_Centrality
from .Internal_Evaluation.Synthetic import simulated_ts_stl, synthetic_anomaly_injection_type
from .Internal_Evaluation.RA import run_rank_agg

from .Ensembling.OE import run_outlier_ens
from .Ensembling.HITS import HITS
from .Ensembling.IOE import Unsupervised_ens
from .Ensembling.SELECT.baseline_select import Select

from .Generation.Aug import Label_Aug, majority_voting
from .Generation.Clean import Label_Clean
from .Generation.UADB.pipeline import Pipeline
from .Generation.UADB.config import Config

from .Meta_learning.Pretraining_pipeline.Step_1_Create_Window_Dataset import split_ts, extract_catch22_features

MS_Pool = ['CQ', 'UEC', 'MC', 'Synthetic', 'TSADAMS']
MG_Pool = ['OE', 'SELECT', 'IOE', 'HITS', 'AutoOD_A', 'AutoOD_C', 'UADB', 'AutoTSAD',
           'SATzilla', 'ISAC', 'ARGOSMART', 'MetaOD', 'MSAD', 'UReg', 'CFact',
           'Oracle', 'SS', 'Random']

MS_Agg_Pool = ['CQ_XBS', 'CQ_STD', 'CQ_R2', 'CQ_Hubert', 'CQ_CH', 'CQ_Silhouette', 'CQ_I_Index', 'CQ_DB', 'CQ_SD', 'CQ_Dunn', 
               'UEC_EM', 'UEC_MV', 'MC_3', 'MC_5', 'MC_7', 'MC_9', 'MC_12',
               'Synthetic_STL-spikes', 'Synthetic_STL-scale', 'Synthetic_STL-noise', 'Synthetic_STL-cutoff', 'Synthetic_STL-contextual', 'Synthetic_STL-speedup',]

def run_AutoAD(model_name, variant, data, Candidate_Model_Set, args, debug_mode, **kwargs):
    if not debug_mode:
        try:
            function_name = f'run_{model_name}'
            function_to_call = globals()[function_name]
            results = function_to_call(variant, data, Candidate_Model_Set, args, **kwargs)
            return results
        except KeyError:
            error_message = f"Model function '{function_name}' is not defined."
            print(error_message)
            return error_message
        except Exception as e:
            error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
            print(error_message)
            return error_message
    else:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(variant, data, Candidate_Model_Set, args, **kwargs)
        return results        

def run_CQ(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)
    
    Cluster_Quality = run_cluster_quality(variant, det_scores)
    score = det_scores.T[Cluster_Quality.index(min(Cluster_Quality))]
    ranking = np.argsort(np.argsort(np.array(Cluster_Quality)))
    return ranking, score, flag

def run_UEC(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)

    UEC_list = run_unsupervised_evaluation_curves(variant, data, Candidate_Model_Set, args)
    score = det_scores.T[UEC_list.index(min(UEC_list))]
    ranking = np.argsort(np.argsort(np.array(UEC_list)))
    return ranking, score, flag

def run_MC(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    try:
        if precomputed:
            det_scores = loading_scores(Candidate_Model_Set, args)
        MC_list = Model_Centrality(det_scores, n_neighbors=[int(args.variant)])
    except:
        MC_list = random.sample(range(len(Candidate_Model_Set)), len(Candidate_Model_Set))
        flag = False
    score = det_scores.T[MC_list.index(min(MC_list))]
    ranking = np.argsort(np.argsort(np.array(MC_list)))
    return ranking, score, flag

def run_Synthetic(variant, data, Candidate_Model_Set, args, precomputed=False):
    
    if variant.split('-')[0] == 'STL':
        data = simulated_ts_stl(data)

    synthetic_performance_list, flag = synthetic_anomaly_injection_type(data, Candidate_Model_Set, anomaly_type=variant.split('-')[1])
    selected_model_id = np.argmax(synthetic_performance_list)
    selected_model = Candidate_Model_Set[selected_model_id]

    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    ranking = np.argsort(np.argsort(-np.array(synthetic_performance_list)))
    return ranking, score, flag

def run_TSADAMS(variant, data, Candidate_Model_Set, args, precomputed=False):
    flag = True

    ranks = np.zeros((len(MS_Agg_Pool), len(Candidate_Model_Set)))
    for i, ms in enumerate(MS_Agg_Pool):    
        df = pd.read_csv(f'{args.save_dir}{ms}.csv')
        ranks[i, :] = df.loc[df['file']==args.filename, Candidate_Model_Set[0]:Candidate_Model_Set[-1]].to_numpy().squeeze()
    ranks = ranks.astype(int)
    rank_agg = run_rank_agg(variant, ranks)
    ranking = rank_agg.astype(int)

    selected_model = Candidate_Model_Set[np.argmin(ranking)]
    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    return ranking, score, flag


def run_OE(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True
    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)
    score = run_outlier_ens(variant, det_scores)
    return score, flag

def run_OE_tmp(variant, data, Candidate_Model_Set, args, precomputed=True):
    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)
    score = run_outlier_ens(variant, det_scores)
    return score

def run_SELECT(variant, data, Candidate_Model_Set, args, precomputed=True):
    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)
    
    flag = True
    try:
        ranks = scores_to_ranks(det_scores, invert_order=True)
        final_ranks = Select(cut_off=20, iterations=100, n_jobs=6).run(det_scores.T, ranks.T, strategy=variant)
        score = ranks_to_scores(final_ranks)
    except:
        print('Error while running SELECT!')
        flag = False
        score = run_OE_tmp("AVG", data, Candidate_Model_Set, args)

    return score, flag

def run_IOE(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)
    score = Unsupervised_ens(det_scores)
    return score, flag

def run_HITS(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)
    score = HITS(det_scores)
    return score, flag

def run_AutoOD_A(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    pred, all_scores, det_scores, instance_index_ranges, detector_index_ranges = gen_autood_initial_set(data, Candidate_Model_Set, args)    
    try:
        if args.ts_len >= 600000:
            raise ValueError("Time series length exceeds the maximum limit of 600000.")

        if variant == 'Majority_vote':
            score = majority_voting(pred, all_scores, data, instance_index_ranges=instance_index_ranges, detector_index_ranges=detector_index_ranges)
        else:
            classifier_result_list, scores_for_training = Label_Aug(pred, all_scores, data, y=None, instance_index_ranges=instance_index_ranges, 
                                                                    detector_index_ranges=detector_index_ranges, max_iteration=10)
            if variant == 'Orig':
                score = classifier_result_list
            elif variant == 'Ensemble':
                score = np.mean(scores_for_training, axis=1)
    except:
        print('Error while running AutoOD_A!')
        flag = False
        score = run_OE_tmp("AVG", data, Candidate_Model_Set, args)
    return score, flag

def run_AutoOD_C(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    try:
        pred, all_scores, det_scores, instance_index_ranges, detector_index_ranges = gen_autood_initial_set(data, Candidate_Model_Set, args)
        score = Label_Clean(data, data, pred, det_scores, max_iteration=10, initial_set=variant)
    except:
        flag = False
        print('Error while running AutoOD_C!')
        score = run_OE_tmp("AVG", data, Candidate_Model_Set, args)
    return score, flag

def run_UADB(variant, data, Candidate_Model_Set, args, precomputed=True):
    flag = True

    try:
        config = Config()
        config.data_path = os.path.join(args.dataset_dir, args.filename)
        config.experiment_type = variant
        config.initial_labels = run_OE_tmp("AVG", data, Candidate_Model_Set, args)
        config.slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        pipeline = Pipeline(config)
        score = pipeline.boost_co_train()
    except:
        flag = False
        print('Error while running UADB!')
        score = run_OE_tmp("AVG", data, Candidate_Model_Set, args)
    
    return score, flag     


def run_MSAD(variant, data, Candidate_Model_Set, args):

    flag = True

    data = StandardScaler().fit_transform(data)
    data_split = split_ts(data, window_size=1024)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_mat = pd.DataFrame(meta_features)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    if variant == 'ID':
        model_path = f'{args.pretrained_weights}/MSAD/ID.pkl'
    else:
        domain = args.filename.split('_')[4]
        if os.path.exists(f'{args.pretrained_weights}/MSAD/{domain}.pkl'):
            model_path = f'{args.pretrained_weights}/MSAD/{domain}.pkl'
        else:
            model_path = f'{args.pretrained_weights}/MSAD/ID.pkl'
    filename = Path(model_path)
    with open(f'{filename}', 'rb') as input:
        model = pickle.load(input)
    preds = model.predict(meta_mat)
    
    # counter = Counter(preds)
    # selected_model = counter.most_common(1)[0][0]
    # print('selected_model: ', selected_model)
    # score_name = args.filename.split('.')[0]
    # score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')

    counter = Counter(preds)
    score_name = args.filename.split('.')[0]
    selected_model = None
    score = None
    for model, freq in counter.most_common():
        file_path = os.path.join(args.score_dir, model, f"{score_name}.npy")
        if os.path.exists(file_path):
            selected_model = model
            print('selected_model:', selected_model)
            score = np.load(file_path)
            break

    if selected_model is None:
        flag = False
        selected_model = random.choice(Candidate_Model_Set)
        print('selected_model: ', selected_model)
        score_name = args.filename.split('.')[0]
        score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')

    return score, flag

def run_SATzilla(variant, data, Candidate_Model_Set, args):

    data = StandardScaler().fit_transform(data)
    data_split = split_ts(data, window_size=1024)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_mat = pd.DataFrame(meta_features)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    if variant == 'ID':
        model_path = f'{args.pretrained_weights}/SATzilla/ID.pkl'
    else:
        domain = args.filename.split('_')[4]
        if os.path.exists(f'{args.pretrained_weights}/SATzilla/{domain}.pkl'):
            model_path = f'{args.pretrained_weights}/SATzilla/{domain}.pkl'
        else:
            model_path = f'{args.pretrained_weights}/SATzilla/ID.pkl'

    filename = Path(model_path)
    with open(f'{filename}', 'rb') as input:
        model = pickle.load(input)
    preds = model.predict(meta_mat)
    counter = Counter(np.argmax(preds, axis=1))
    most_voted = counter.most_common(1)
    selected_model = Candidate_Model_Set[int(most_voted[0][0])]
    print('selected_model: ', selected_model)

    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    return score, True

def run_UReg(variant, data, Candidate_Model_Set, args):

    data = StandardScaler().fit_transform(data)
    data_split = split_ts(data, window_size=1024)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_mat = pd.DataFrame(meta_features)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    if variant == 'ID':
        model_path = f'{args.pretrained_weights}/UReg/ID.pkl'
    else:
        domain = args.filename.split('_')[4]
        if os.path.exists(f'{args.pretrained_weights}/UReg/{domain}.pkl'):
            model_path = f'{args.pretrained_weights}/UReg/{domain}.pkl'
        else:
            model_path = f'{args.pretrained_weights}/UReg/ID.pkl'

    filename = Path(model_path)
    with open(f'{filename}', 'rb') as input:
        result = pickle.load(input)
    U_pred = np.column_stack([rf.predict(meta_mat) for rf in result['models']])
    prediction_scores = U_pred.dot(result['DVt'])
    preds = np.argmax(prediction_scores, axis=1)
    counter = Counter(preds)
    most_voted = counter.most_common(1)
    selected_model = Candidate_Model_Set[int(most_voted[0][0])]
    print('selected_model: ', selected_model)

    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    return score, True

def run_CFact(variant, data, Candidate_Model_Set, args):

    data = StandardScaler().fit_transform(data)
    data_split = split_ts(data, window_size=1024)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_mat = pd.DataFrame(meta_features)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    if variant == 'ID':
        model_path = f'{args.pretrained_weights}/CFact/ID.pkl'
    else:
        domain = args.filename.split('_')[4]
        if os.path.exists(f'{args.pretrained_weights}/CFact/{domain}.pkl'):
            model_path = f'{args.pretrained_weights}/CFact/{domain}.pkl'
        else:
            model_path = f'{args.pretrained_weights}/CFact/ID.pkl'

    filename = Path(model_path)
    with open(f'{filename}', 'rb') as input:
        result = pickle.load(input)

    predictions = []
    for internal_recommender in result['models']:
        U_pred = np.column_stack([rf.predict(meta_mat) for rf in internal_recommender['models']])
        prediction_scores = U_pred.dot(internal_recommender['DVt'])
        predictions.append(pd.DataFrame(prediction_scores, columns=internal_recommender['configurations'].tolist()))
    predictions_concat = pd.concat(predictions, axis=1)            
    preds = np.argmax(predictions_concat.to_numpy(), axis=1)
    
    counter = Counter(preds)
    most_voted = counter.most_common(1)
    selected_model = Candidate_Model_Set[int(most_voted[0][0])]
    print('selected_model: ', selected_model)

    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    return score, True

def run_ARGOSMART(variant, data, Candidate_Model_Set, args):

    flag = True

    if data.shape[1] == 1:
        file_list = '/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-U-Label.csv'
        feature_list = '/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_U.csv'
    else:
        file_list = '/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-M-Label.csv'
        feature_list = '/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_M.csv'        
    domain = args.filename.split('_')[4]

    if variant == 'ID':
        train_val_set = pd.read_csv(file_list)['file_name'].values.tolist()
    else:
        df = pd.read_csv(file_list)
        train_val_set = df[df["domain_name"] != domain]['file_name'].values.tolist()

    data_feature = pd.read_csv(feature_list, index_col=0)
    data_feature['max_id'] = data_feature.iloc[:,:len(Candidate_Model_Set)].idxmax(axis=1)
    train_val_set_base = set(f.split('.')[0] for f in train_val_set)
    train_val_data = data_feature.loc[data_feature.index.str.split('_').str[:-1].str.join('_').isin(train_val_set_base)]
    X_tr_val, y_tr_val = train_val_data.iloc[:, len(Candidate_Model_Set):-1], train_val_data.iloc[:, -1]
    X_tr_val = X_tr_val.replace([np.nan, np.inf, -np.inf], 0)

    data = StandardScaler().fit_transform(data)
    data_split = split_ts(data, window_size=1024)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_mat = pd.DataFrame(meta_features)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_tr_val)
    preds = []
    for win in range(meta_mat.shape[0]):
        distances, indices = nbrs.kneighbors(meta_mat.iloc[win, :].values.reshape(1, -1))
        preds.append(y_tr_val[indices[0][0]])
    
    # counter = Counter(preds)
    # selected_model = counter.most_common(1)[0][0]
    # print('selected_model: ', selected_model)
    # score_name = args.filename.split('.')[0]
    # score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')

    counter = Counter(preds)
    score_name = args.filename.split('.')[0]
    selected_model = None
    score = None
    for model, count in counter.most_common():
        file_path = os.path.join(args.score_dir, str(model), f"{score_name}.npy")
        if os.path.exists(file_path):
            selected_model = model
            print('selected_model: ', selected_model)
            score = np.load(file_path)
            break

    if selected_model is None:
        flag = False
        selected_model = random.choice(Candidate_Model_Set)
        print('selected_model: ', selected_model)
        score_name = args.filename.split('.')[0]
        score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')

    return score, flag

def run_MetaOD(variant, data, Candidate_Model_Set, args):
    import sys
    sys.path.append('/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline')

    data = StandardScaler().fit_transform(data)
    data_split = split_ts(data, window_size=1024)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_mat = pd.DataFrame(meta_features)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    if variant == 'ID':
        model_path = f'{args.pretrained_weights}/MetaOD/ID.joblib'
    else:
        domain = args.filename.split('_')[4]
        if os.path.exists(f'{args.pretrained_weights}/MetaOD/{domain}.joblib'):
            model_path = f'{args.pretrained_weights}/MetaOD/{domain}.joblib'
        else:
            model_path = f'{args.pretrained_weights}/MetaOD/ID.joblib'

    clf = load(model_path)
    metaod_predict = clf.predict(meta_mat)
    ranks = np.argsort(metaod_predict)
    ranks = np.argsort(ranks)
    preds_agg = np.sum(ranks, axis=0)
    preds = np.argsort(preds_agg, axis=0)[::-1][:1]
    selected_model = Candidate_Model_Set[preds[0]]
    print('selected_model: ', selected_model)

    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    return score, True

def run_ISAC(variant, data, Candidate_Model_Set, args):

    flag = True

    data = StandardScaler().fit_transform(data)
    data_split = split_ts(data, window_size=1024)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_mat = pd.DataFrame(meta_features)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    if variant == 'ID':
        model_path = f'{args.pretrained_weights}/ISAC/ID.joblib'
        cluster_dict_load = np.load(f'{args.pretrained_weights}/ISAC/ID_cluster_dict.npy', allow_pickle=True).item()
    else:
        domain = args.filename.split('_')[4]

        if os.path.exists(f'{args.pretrained_weights}/ISAC/{domain}.joblib'):
            model_path = f'{args.pretrained_weights}/ISAC/{domain}.joblib'
            cluster_dict_load = np.load(f'{args.pretrained_weights}/ISAC/{domain}_cluster_dict.npy', allow_pickle=True).item()
        else:
            model_path = f'{args.pretrained_weights}/ISAC/ID.joblib'
            cluster_dict_load = np.load(f'{args.pretrained_weights}/ISAC/ID_cluster_dict.npy', allow_pickle=True).item()

    clf = load(model_path)
    all_preds = []
    for win in range(meta_mat.shape[0]):
        preds = clf.predict(meta_mat)
        all_preds.append(Candidate_Model_Set.index(cluster_dict_load[preds[0]]))
    
    # counter = Counter(all_preds)
    # most_voted = counter.most_common(1)
    # selected_model = Candidate_Model_Set[int(most_voted[0][0])]
    # print('selected_model: ', selected_model)
    # score_name = args.filename.split('.')[0]
    # score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')

    counter = Counter(all_preds)
    score_name = args.filename.split('.')[0]
    selected_model = None
    score = None
    for candidate_idx, count in counter.most_common():
        candidate_model = Candidate_Model_Set[int(candidate_idx)]
        file_path = os.path.join(args.score_dir, candidate_model, f"{score_name}.npy")
        if os.path.exists(file_path):
            selected_model = candidate_model
            print('selected_model: ', selected_model)
            score = np.load(file_path)
            break

    if selected_model is None:
        flag = False
        selected_model = random.choice(Candidate_Model_Set)
        print('selected_model: ', selected_model)
        score_name = args.filename.split('.')[0]
        score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')

    return score, flag

def run_Oracle(variant, data, Candidate_Model_Set, args):

    file_metric_dict = {}          
    for det in Candidate_Model_Set:
        df = pd.read_csv(f'{args.eval_path}/{det}.csv')
        line = df.loc[df['file'] == args.filename]
        Metric = line[f'VUS-PR'].values[0]
        file_metric_dict[det] = Metric
    selected_model = max(file_metric_dict, key=file_metric_dict.get)
    print('selected_model: ', selected_model)

    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    return score, True

def run_SS(variant, data, Candidate_Model_Set, args):

    domain = args.filename.split('_')[4]
    select_filelist_read = pd.read_csv(args.eval_list)
    select_filelist = select_filelist_read[select_filelist_read['domain_name']==domain]

    det_performance_dict = {det: 0 for det in Candidate_Model_Set}
    for index, row in select_filelist.iterrows():
        for det in Candidate_Model_Set:
            df = pd.read_csv(f"{args.eval_path}/{det}.csv")
            line = df.loc[df['file'] == row['file_name']]
            Metric = line[f'VUS-PR'].values[0]
            det_performance_dict[det] += Metric
    sorted_dict = dict(sorted(det_performance_dict.items(), key=lambda item: item[1], reverse=True))
    selected_model = list(sorted_dict.keys())[0]
    print('selected_model: ', selected_model)

    score_name = args.filename.split('.')[0]
    score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
    return score, True


def run_AutoTSAD(variant, data, Candidate_Model_Set, args):

    flag = True
    result_dir = '/data/liuqinghua/code/ts/TSAD-AutoML/run_baseline/AutoTSAD/result_AutoTSAD_baseline'

    try:
        tgt_path = os.path.join(result_dir, args.filename, '**', 'rankings', '*')
        score_paths = glob.glob(tgt_path, recursive=True)
        if not score_paths:
            raise FileNotFoundError(f"No matching directories found for: {tgt_path}")
        
        rankings_dir = score_paths[0]
        expected_file = os.path.join(rankings_dir, 'combined-score.csv')
        if not os.path.exists(expected_file):
            raise FileNotFoundError(f"Expected file not found: {expected_file}")
        score = pd.read_csv(expected_file, header=None).to_numpy().ravel()
    except:
        flag = False
        print('Error while running AutoTSAD!')
        score = run_OE_tmp("AVG", data, Candidate_Model_Set, args)
    
    return score, flag   

def run_Random_TS(variant, data, Candidate_Model_Set, args):
    pass

def run_Random_Dataset(data, Candidate_Model_Set, args):
    pass