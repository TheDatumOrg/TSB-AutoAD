#  This function is adapted from [MSAD] by [boniolp]
#  Original source: [https://github.com/boniolp/MSAD]

import argparse
import os
from time import perf_counter
import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

names = {
        "knn": "Nearest Neighbors",
        "svc_linear": "Linear SVM",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "mlp": "Neural Net",
        "ada_boost": "AdaBoost",
        "bayes": "Naive Bayes",
        "qda": "QDA",
}

classifiers = {
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "svc_linear": LinearSVC(C=0.025, verbose=True),
        "decision_tree": DecisionTreeClassifier(max_depth=5),
        "random_forest": RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=4, verbose=True),
        "mlp": MLPClassifier(alpha=1, max_iter=1000, verbose=True),
        "ada_boost": AdaBoostClassifier(),
        "bayes": GaussianNB(),
        "qda": QuadraticDiscriminantAnalysis(),
}

# Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
#         'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
#         'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']
# Candidate_Model_Set = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
#                     'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']

Candidate_Model_Set = ['Sub_PCA', 'TimesFM', 'POLY', 'Series2Graph', 'MOMENT_FT', 'MOMENT_ZS', 'KMeansAD_U', 'USAD', 'Sub_KNN', 'MatrixProfile']
# Candidate_Model_Set = ['CNN', 'OmniAnomaly', 'PCA', 'LSTMAD', 'USAD', 'AutoEncoder', 'KMeansAD', 'CBLOF', 'MCD', 'OCSVM']


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

    # Select the classifier
    classifier = classifiers[classifier_name]
    clf_name = classifier_name

    # For svc_linear use only a random subset of the dataset to train
    if 'svc' in classifier_name and len(y_train) > 200000:
        rand_ind = np.random.randint(low=0, high=len(y_train), size=200000)
        X_train = X_train.iloc[rand_ind]
        y_train = y_train.iloc[rand_ind]

    # Fit the classifier
    print(f'----------------------------------\nTraining {names[classifier_name]}...')
    tic = perf_counter()
    classifier.fit(X_train, y_train)
    toc = perf_counter()

    # Print training time
    training_time = toc - tic
    print(f"training time: {training_time} secs")

    # Save
    filename = Path(os.path.join(path_save, f"{domain}.pkl"))
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'wb') as output:
        pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_U_sub.csv')
    parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-U-Label.csv')
    parser.add_argument('-ps', '--path_save', type=str, 
                default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights_sub/TSB-AD-U/MSAD")
    # parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/data/TSB_meta_feature_M_sub.csv')
    # parser.add_argument('-f', '--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-M-Label.csv')
    # parser.add_argument('-ps', '--path_save', type=str, 
    #             default="/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/TSB_AutoAD/Meta_learning/Pretraining_pipeline/weights_sub/TSB-AD-M/MSAD")

    parser.add_argument('-c', '--classifier', type=str, default='random_forest')
    parser.add_argument('-v', '--validation', type=bool, default=False)
    parser.add_argument('-t', '--domain', type=str, default='ID', help='ID, WebService, Medical, Facility, Synthetic, HumanActivity, Sensor, Environment, Finance, Traffic')

    args = parser.parse_args()

    # Option to all classifiers
    if args.classifier == 'all':
        clf_list = list(classifiers.keys())
    else:
        clf_list = [args.classifier]

    for classifier in clf_list:
        meta_train(
            data_path=args.path,
            file_list=args.file_list,
            classifier_name=classifier,
            validation=args.validation,
            path_save=args.path_save,
            domain=args.domain
        )
