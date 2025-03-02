import argparse
import os
from time import perf_counter
import re
from datetime import datetime
import numpy as np
import pandas as pd
from pyclustering.cluster.gmeans import gmeans
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from joblib import dump, load


# Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
#         'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
#         'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']
Candidate_Model_Set = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
                    'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']


def meta_train(data_path, file_list, candidate_eval_list, validation, path_save=None, domain=None):
    
    # Read Train/Val/Test splits read from file
    if domain == 'ID':
        meta_train_list = pd.read_csv(file_list)['file_name'].values.tolist()
    else:
        df = pd.read_csv(file_list)
        if domain in df["domain_name"].unique():
            meta_train_list = df[df["domain_name"] != domain]['file_name'].values.tolist()
        else:
            print(f'No {domain}')
            exit()

    if validation:
        train_set, val_set = train_test_split(meta_train_list, test_size=0.2, random_state=2024)
    else:
        train_set = meta_train_list
        val_set = []  
    # print(train_set[:5])    # ['003_NAB_id_3_WebService_tr_1362_1st_1462.csv']

    # Read tabular data
    data = pd.read_csv(data_path, index_col=0)
    data['max_id'] = data.iloc[:,:len(Candidate_Model_Set)].idxmax(axis=1)
    # print(data.index[:5])    # ['003_NAB_id_3_WebService_tr_1362_1st_1462_0']

    # Ensure matching based on partial filename match
    train_set_base = set(f.split('.')[0] for f in train_set)
    val_set_base = set(f.split('.')[0] for f in val_set)
    training_data = data.loc[data.index.str.split('_').str[:-1].str.join('_').isin(train_set_base)]
    val_data = data.loc[data.index.str.split('_').str[:-1].str.join('_').isin(set(val_set_base))]

    # print(training_data)
    # print(training_data.shape)

    # Split data from labels
    X_train, X_val = training_data.iloc[:, len(Candidate_Model_Set):-1], val_data.iloc[:, len(Candidate_Model_Set):-1]
    y_train, y_val = training_data.iloc[:, -1], val_data.iloc[:, -1]
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)

    X_train = X_train.replace([np.nan, np.inf, -np.inf], 0)
    X_val = X_val.replace([np.nan, np.inf, -np.inf], 0)

    merged_pd = pd.concat([X_train, X_val], axis=0)

    print('merged_pd: ', merged_pd)

    X_tr_val = merged_pd.to_numpy().astype('float64')
    X_tr_val[np.isinf(X_tr_val)] = np.nan
    X_tr_val[np.isnan(X_tr_val)] = np.nanmean(X_tr_val)

    # X_tr_val = X_tr_val[:100,]

    data_index = list(merged_pd.index)

    gmeans_instance = gmeans(X_tr_val, random_state=2024).process()

    filename = Path(os.path.join(path_save, f"{domain}.joblib"))
    filename.parent.mkdir(parents=True, exist_ok=True)    
    dump(gmeans_instance, f'{path_save}/{domain}.joblib')

    clusters = gmeans_instance.get_clusters()
    cluster_dict = {i: 0 for i in range(len(clusters))}
    for i, cluster in enumerate(clusters):
        det_performance_dict = {det: 0 for det in Candidate_Model_Set}
        for id in cluster:
            file_name_list = []
            file_name = data_index[id].rsplit('_', 1)[0] + '.csv'
            file_name_list.append(file_name)
        for file_name in set(file_name_list):
            for det in Candidate_Model_Set:
                df = pd.read_csv(f"{candidate_eval_list}{det}.csv")
                metric_value = df.loc[df['file']==file_name]['VUS-PR'].values[0]                 
                det_performance_dict[det] += metric_value   
        det_cluster = max(det_performance_dict, key=det_performance_dict.get)
        cluster_dict[i] = det_cluster
    np.save(f'{path_save}/{domain}_cluster_dict.npy', cluster_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_U.csv')
    # parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-U-Label.csv')
    # parser.add_argument('--candidate_eval_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/benchmark_exp/eval/Candidate/TSB-AD-U/')
    # parser.add_argument('-ps', '--path_save', type=str, 
    #             default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights/TSB-AD-U/ISAC")

    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_M.csv')
    parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-M-Label.csv')
    parser.add_argument('--candidate_eval_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/benchmark_exp/eval/Candidate/TSB-AD-M/')
    parser.add_argument('-ps', '--path_save', type=str, 
                default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights/TSB-AD-M/ISAC")


    parser.add_argument('-v', '--validation', type=bool, default=False)
    parser.add_argument('-t', '--domain', type=str, default='ID', help='ID, WebService, Medical, Facility, Synthetic, HumanActivity, Sensor, Environment, Finance, Traffic')

    args = parser.parse_args()

    meta_train(
        data_path=args.path,
        file_list=args.file_list,
        candidate_eval_list=args.candidate_eval_list,
        validation=args.validation,
        path_save=args.path_save,
        domain=args.domain
    )
