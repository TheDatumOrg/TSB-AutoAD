import argparse
import numpy as np
import pandas as pd
from Metaod_core import MetaODClass
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import os

# Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
#         'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
#         'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']
Candidate_Model_Set = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
                    'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']


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

    train_set = y_train.to_numpy().astype('float64')
    valid_set = y_val.to_numpy().astype('float64')

    train_meta = X_train.to_numpy().astype('float64')
    valid_meta = X_val.to_numpy().astype('float64')

    meta_scalar = MinMaxScaler()
    train_meta = meta_scalar.fit_transform(train_meta)
    valid_meta = meta_scalar.fit_transform(valid_meta)

    train_meta[np.isinf(train_meta)] = np.nan
    train_meta[np.isnan(train_meta)] = np.nanmean(train_meta)
    valid_meta[np.isinf(valid_meta)] = np.nan
    valid_meta[np.isnan(valid_meta)] = np.nanmean(valid_meta)

    clf = MetaODClass(train_set, valid_performance=valid_set, n_factors=10,
                    learning='sgd')
    clf.train(meta_features=train_meta, valid_meta=valid_meta, n_iter=10,
            learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1, n_steps=8)
    print('Done training')

    # Save
    filename = Path(os.path.join(path_save, f"{domain}.joblib"))
    filename.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, f'{path_save}/{domain}.joblib')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_U.csv')
    # parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-U-Label.csv')
    # parser.add_argument('-ps', '--path_save', type=str, 
    #             default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights/TSB-AD-U/MetaOD")
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_M.csv')
    parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-M-Label.csv')
    parser.add_argument('-ps', '--path_save', type=str, 
                default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights/TSB-AD-M/MetaOD")

    parser.add_argument('-c', '--classifier', type=str, default='random_forest')
    parser.add_argument('-v', '--validation', type=bool, default=True)
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
