import pandas as pd
import numpy as np
import argparse, time, os
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
        'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
        'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Eva')
    parser.add_argument('--dataset_dir', type=str, default='/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-U/')
    parser.add_argument('--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/testbed/file_list/TSB-AD-U.csv')
    parser.add_argument('--score_dir', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/resource/score/uni/')
    parser.add_argument('--save_dir', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/benchmark_exp/eval/Candidate/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='Sub_IForest')
    parser.add_argument('--resume', type=bool, default=True)
    args = parser.parse_args()

    ## Resume
    if args.resume and os.path.exists(f'{args.save_dir}/{args.AD_Name}.csv'):
        df = pd.read_csv(f'{args.save_dir}/{args.AD_Name}.csv')
        write_csv =  df.values.tolist()
        Resume_file_name_list = df['file'].tolist()
    else:
        write_csv = []
        Resume_file_name_list = []

    file_list = pd.read_csv(args.file_list)['file_name'].values
    for filename in file_list:
        if filename in Resume_file_name_list: continue
        print('Evaluating:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        # print('data: ', data.shape)
        # print('label: ', label.shape)

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)

        try:
            if args.AD_Name == 'Random_C':
                anomaly_score_pool = []
                for i in range(5):
                    anomaly_score_pool.append(np.random.uniform(size=data.shape[0]))
                anomaly_score = np.mean(np.array(anomaly_score_pool), axis=0)
            elif args.AD_Name == 'Random_D':
                anomaly_score_pool = []
                for i in range(5):
                    anomaly_score_pool.append((np.random.uniform(size=data.shape[0]) > 0.5).astype(float))
                anomaly_score = np.mean(np.array(anomaly_score_pool), axis=0)
            elif args.AD_Name == 'Oracle':
                anomaly_score = label
            else: 
                print(f'---{args.AD_Name}---')
                anomaly_score = np.load(f'{args.score_dir}{args.AD_Name}/{filename[:-4]}.npy')

            if len(anomaly_score) < len(label):
                pad_length = len(label) - len(anomaly_score)
                anomaly_score = np.pad(anomaly_score, (0, pad_length), 'constant', constant_values=(0, anomaly_score[-1]))

            evaluation_result = get_metrics(anomaly_score, label, slidingWindow=slidingWindow)
            print('evaluation_result: ', evaluation_result)
            list_w = list(evaluation_result.values())
        except:       
            list_w = [0]*9
        run_time = "unknown"
        list_w.insert(0, run_time)
        list_w.insert(0, filename)
        write_csv.append(list_w)

        ## Temp Save
        col_w = list(evaluation_result.keys())
        col_w.insert(0, 'Time')
        col_w.insert(0, 'file')
        w_csv = pd.DataFrame(write_csv, columns=col_w)
        w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)