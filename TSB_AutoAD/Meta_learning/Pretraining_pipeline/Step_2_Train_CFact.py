#  This function is adapted from [ADRecommender] by [Jose Manuel Navarro et al.]
#  Original source: [https://figshare.com/articles/code/Meta-Learning_for_Fast_Model_Recommendation_in_Unsupervised_Multivariate_Time_Series_Anomaly_Detection/22320367]

import argparse
import os
from time import perf_counter
import re
from datetime import datetime
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path



# Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
#         'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
#         'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']
Candidate_Model_Set = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
                    'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']

def fit_u_regression(train_performance, train_metafeatures, n_factors=5):
    # Adjust n_factors if necessary
    n_factors = min(n_factors, train_performance.shape[1])
    
    # SVD Decomposition
    svd = TruncatedSVD(n_components=n_factors)
    U = svd.fit_transform(train_performance)
    D = np.diag(svd.singular_values_)
    Vt = svd.components_
    DVt = D.dot(Vt)
    
    # Train a Random Forest for each factor
    models = []
    for i in range(n_factors):
        y = U[:, i]
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, verbose=True)
        rf.fit(train_metafeatures, y)
        models.append(rf)
    
    result = {
        'models': models,
        'DVt': DVt,
        'configurations': train_performance.columns,
        'recommender_type': 'URegression (RF)'
    }    
    return result

def fit_cfact(train_performance, train_metafeatures, n_factors=5):
    # UMAP projection
    umap = UMAP(n_components=10, metric='manhattan')
    algo_proj = umap.fit_transform(train_performance.T)
    
    # Clustering
    algo_clust = DBSCAN(eps=1).fit(algo_proj).labels_

    # Fit UReg model for each cluster
    models = []
    unique_clusters = np.unique(algo_clust)
    for clust_num in unique_clusters:
        indices = np.where(algo_clust == clust_num)[0]

        subset_performance = train_performance.iloc[:, indices]
        # subset_metafeatures = train_metafeatures[:, indices]

        print('subset_performance: {}'.format(subset_performance.shape))

        try:
            u_reg = fit_u_regression(subset_performance, train_metafeatures, n_factors)
        except Exception as e:
            print('Detected error with the internal SVD computations. Trying again.')
            continue
        models.append(u_reg)

    result = {
        'models': models,
        'cluster_indices': algo_clust,
        'configurations': train_performance.columns,
        'recommender_type': 'cFact'
    }
    
    return result

def meta_train(data_path, file_list, classifier_name, validation, path_save=None, domain=None):
    
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
    # print(data.index[:5])    # ['003_NAB_id_3_WebService_tr_1362_1st_1462_0']

    # Ensure matching based on partial filename match
    train_set_base = set(f.split('.')[0] for f in train_set)
    val_set_base = set(f.split('.')[0] for f in val_set)
    training_data = data.loc[data.index.str.split('_').str[:-1].str.join('_').isin(train_set_base)]
    val_data = data.loc[data.index.str.split('_').str[:-1].str.join('_').isin(set(val_set_base))]

    # print(training_data)
    # print(training_data.shape)

    # Split data from labels
    X_train, X_val = training_data.iloc[:, len(Candidate_Model_Set):], val_data.iloc[:, len(Candidate_Model_Set):]
    y_train, y_val = training_data.iloc[:, :len(Candidate_Model_Set)], val_data.iloc[:, :len(Candidate_Model_Set)]
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)

    X_train = X_train.replace([np.nan, np.inf, -np.inf], 0)
    X_val = X_val.replace([np.nan, np.inf, -np.inf], 0)


    # Fit the classifier
    tic = perf_counter()
    result = fit_cfact(y_train, X_train)
    toc = perf_counter()

    # Print training time
    training_time = toc - tic
    print(f"training time: {training_time} secs")

    # Save
    filename = Path(os.path.join(path_save, f"{domain}.pkl"))
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'wb') as output:
        pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_U.csv')
    # parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-U-Label.csv')
    # parser.add_argument('-ps', '--path_save', type=str, 
    #             default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights/TSB-AD-U/CFact")

    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_M.csv')
    parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-M-Label.csv')
    parser.add_argument('-ps', '--path_save', type=str, 
                default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights/TSB-AD-M/CFact")

    parser.add_argument('-c', '--classifier', type=str, default=None)
    parser.add_argument('-v', '--validation', type=bool, default=False)
    parser.add_argument('-t', '--domain', type=str, default='ID', help='ID, WebService, Medical, Facility, Synthetic, HumanActivity, Sensor, Environment, Finance, Traffic')
    args = parser.parse_args()

    meta_train(
        data_path=args.path,
        file_list=args.file_list,
        classifier_name=args.classifier,
        validation=args.validation,
        path_save=args.path_save,
        domain=args.domain
    )
