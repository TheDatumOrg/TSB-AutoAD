import sys
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from pycatch22 import catch22_all

# Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
#         'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
#         'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']
# Candidate_Model_Set = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
#                     'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']

# Candidate_Model_Set = ['Sub_PCA', 'TimesFM', 'POLY', 'Series2Graph', 'MOMENT_FT', 'MOMENT_ZS', 'KMeansAD_U', 'USAD', 'Sub_KNN', 'MatrixProfile']
Candidate_Model_Set = ['CNN', 'OmniAnomaly', 'PCA', 'LSTMAD', 'USAD', 'AutoEncoder', 'KMeansAD', 'CBLOF', 'MCD', 'OCSVM']

def extract_catch22_features(data):
    if data.shape[1] == 1:  # Univariate case
        features = np.array([catch22_all(data[:, 0])['values']])  # Shape (1,22)
    else:  # Multivariate case
        features = np.array([catch22_all(data[:, i])['values'] for i in range(data.shape[1])])  # Shape (channels, 22)
        summary_features = np.vstack([
            np.min(features, axis=0),
            np.percentile(features, 25, axis=0),
            np.mean(features, axis=0),
            np.percentile(features, 75, axis=0),
            np.max(features, axis=0)
        ])  # Shape (5, 22)
        features = summary_features.flatten().reshape(1, -1)  # Shape (1, 110)
    
    # Replace NaN values with 0
    features = np.nan_to_num(features)
    return features

def create_tmp_dataset(file_list_path, dataset_dir, save_dir, window_size, metric_path, metric):
    file_list = pd.read_csv(file_list_path)['file_name'].values
    all_data = []

    for filename in file_list:
        print('filename: ', filename)
        file_path = os.path.join(dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)

        data = StandardScaler().fit_transform(data)
        data_split = split_ts(data, window_size)
        meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
        
        candidate_metric_list = []
        for det in Candidate_Model_Set:
            df = pd.read_csv(metric_path+det+'.csv')
            value = df[df['file'] == filename][metric].values[0]
            candidate_metric_list.append(value)
        
        candidate_metric = np.array(candidate_metric_list).squeeze()
        candidate_metric = np.expand_dims(candidate_metric, 0).repeat(meta_features.shape[0], axis=0)
        data_merge = np.concatenate((candidate_metric, meta_features), axis=1)

        new_names = [filename.split('.')[0] + '_{}'.format(i) for i in range(meta_features.shape[0])]
        col_names = Candidate_Model_Set + ["val_{}".format(i) for i in range(meta_features.shape[1])]

        df = pd.DataFrame(data_merge, index=new_names, columns=col_names)
        all_data.append(df)
    
    merged_df = pd.concat(all_data)
    merged_df.to_csv(save_dir, index=True)

def split_ts(data, window_size):
    if data.shape[0] < window_size:
        pad_length = window_size - data.shape[0]
        data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        return np.expand_dims(data, axis=0)

    modulo = data.shape[0] % window_size
    k = data[modulo:].shape[0] / window_size
    assert(math.ceil(k) == k)
    data_split = np.array(np.split(data[modulo:], int(k)))
    if modulo != 0:
        first_window = data[:window_size].reshape(1, window_size, -1)
        data_split = np.vstack((first_window, data_split))
    return data_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating Window Dataset')
    # parser.add_argument('--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-U-Label.csv')
    # parser.add_argument('--dataset_dir', type=str, default='/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-U/')
    # parser.add_argument('--save_dir', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_U_sub.csv')
    # parser.add_argument('--metric_path', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/benchmark_exp/eval/Candidate/TSB-AD-U/')

    parser.add_argument('--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-M-Label.csv')
    parser.add_argument('--dataset_dir', type=str, default='/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-M/')
    parser.add_argument('--save_dir', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_M_sub.csv')
    parser.add_argument('--metric_path', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/benchmark_exp/eval/Candidate/TSB-AD-M/')

    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--metric', type=str, default='VUS-PR', help='VUS-PR, AUC-PR')
    args = parser.parse_args()
    create_tmp_dataset(
        file_list_path=args.file_list,
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir, 
        window_size=args.window_size,
        metric_path=args.metric_path,
        metric=args.metric
    )
