from pathlib import Path
from periodicity_detection import number_peaks
import numpy as np
import sys
sys.path.append('/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/AutoTSAD/Internal_Evaluation')
from limiting import TimeseriesLimiter
from pruning import filter_similar_timeseries
from dataset import TestDataset, TrainingDatasetCollection
from regiming import get_regime_masks
from cleaning import clean_base_timeseries

def analyze_dataset(testdataset):
    # Use a simpler method for period detection:
    periods = [number_peaks(testdataset.data, n=100)]
    return periods

def generate_base_ts_collection(testdataset):
    periods = analyze_dataset(testdataset)

    train_collection = TrainingDatasetCollection.from_base_timeseries(testdataset)

    # segment base time series based on prominent behavior per period size
    split_into_regimes(train_collection, periods)

    if len(train_collection) == 0:
        print("No proper period_size found! Extracting samples with period_size 100 as base TSs!")
        TimeseriesLimiter().extract_sampled_regime(train_collection, period_size=100, in_place=True)

    # limit number of similar base time series
    filter_similar_timeseries(train_collection)

    return train_collection

def split_into_regimes(train_collection, period_sizes):

    # capture queue object for subprocesses
    def call_get_regime_masks(train_collection, period_size):

        masks = get_regime_masks(train_collection, period_size)
        periods = np.repeat(period_size, masks.shape[0])
        return masks, periods

    results = []
    for period_size in period_sizes:
        # Call the function and append the result
        result = call_get_regime_masks(train_collection, period_size)
        results.append(result)

    generated_masks, periods = zip(*results)
    generated_masks = np.concatenate(generated_masks, axis=0)
    periods = np.concatenate(periods, axis=0)

    for i in np.arange(generated_masks.shape[0]):
        train_collection.add_base_ts(generated_masks[i], period_size=periods[i])

    return train_collection

def synthetic_data_gen(file_path):

    testdataset = TestDataset.from_file(Path(file_path))
    print("\n############################################################")
    print("#             STEP 1: Training data generation             #")
    print("############################################################")

    print("\n# Generating base time series")
    print("###########################")
    train_collection = generate_base_ts_collection(testdataset)

    print("\n# Cleaning base time series")
    print("###########################")
    train_collection = clean_base_timeseries(train_collection, use_gt=False)

    print("\n# Limiting base time series")
    train_collection = limit_base_timeseries(train_collection)

    print("\n# Injecting anomalies")
    print("###########################")
    train_collection = generate_anomaly_time_series(train_collection)

    print("Finished generating training data.\n")
    return train_collection

if __name__ == '__main__':
    file_path = '/data/liuqinghua/code/ts/TSAD-AutoML/TSB-AutoAD/data/TSB-UAD-timeeval/NAB/NAB_data_art0_2.out'
    train_collection = synthetic_data_gen(file_path)