import numpy as np
from .inject_anomalies import InjectAnomalies
from .inject_anomalies import gen_synthetic_performance_list
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.seasonal import STL
from TSB_AD.utils.slidingWindows import find_length_rank
from ..utils import downsample_ts

def simulated_ts_stl(data, slidingWindow=None):
    """
    Applies STL decomposition to each dimension of a multivariate time series 
    and reconstructs a simulated version by adding noise to the residual component.
    
    Parameters:
    - data: numpy array of shape (T, D), where T is the number of time points and D is the number of variables.
    - slidingWindow: int, optional. If None, it is determined automatically for each dimension.
    
    Returns:
    - simulated_data: numpy array of shape (T, D)
    """
    T, D = data.shape
    simulated_data = np.zeros_like(data)
    
    for dim in range(D):
        dim_data = data[:, dim]
        
        # Determine period dynamically if not provided
        if slidingWindow is None:
            slidingWindow_dim = find_length_rank(dim_data.reshape(-1, 1), rank=1)
        else:
            slidingWindow_dim = slidingWindow
        
        # Apply STL decomposition
        stl = STL(dim_data, period=slidingWindow_dim)
        result = stl.fit()
        seasonal, trend, resid = result.seasonal, result.trend, result.resid
        
        # Generate simulated data
        noise = np.random.normal(np.mean(resid), np.std(resid), len(dim_data))
        simulated_data[:, dim] = seasonal + noise
    
    return simulated_data

def synthetic_anomaly_injection_type(data, Det_pool, anomaly_type):

    flag = True

    data_ds = downsample_ts(data, rate=10)
    T = data_ds.T   #  (feature, ts)
    # T = T[:10000]

    data_std = max(np.std(T), 0.01)
    synthetic_time_series = []
    synthetic_anomaly_labels = []

    anomaly_obj = InjectAnomalies(random_state=0,
                                verbose=False,
                                max_window_size=128,
                                min_window_size=8)
    for anomaly_params in list(
            ParameterGrid(ANOMALY_PARAM_GRID[anomaly_type])):
        anomaly_params['T'] = T
        anomaly_params['scale'] = anomaly_params['scale'] * data_std
        anomaly_params['anomaly_type'] = anomaly_type
        # print('anomaly_params: ', anomaly_params)

        # Inject synthetic anomalies to the data
        inject_label = True
        try:
            T_a, anomaly_sizes, anomaly_labels = anomaly_obj.inject_anomalies(**anomaly_params)            
            synthetic_time_series.append(T_a.T)
            synthetic_anomaly_labels.append(anomaly_labels)
        except:
            flag = False
            inject_label = False
            print('Error while injecting anomaly')

    synthetic_performance_list = []
    if inject_label:
        for i in range(len(synthetic_anomaly_labels)):
            data_i = synthetic_time_series[i]
            label_i = synthetic_anomaly_labels[i]
            try:
                synthetic_performance_i = gen_synthetic_performance_list(data_i, label_i, Det_pool)
            except:
                print('Error while generating synthetic performace list')
                flag = False
                synthetic_performance_i = [0]*len(Det_pool)
            synthetic_performance_list.append(synthetic_performance_i)
    else:
        synthetic_performance_list.append([0]*len(Det_pool))

    synthetic_performance = np.array(synthetic_performance_list).astype(np.float32)
    synthetic_performance_list = np.mean(synthetic_performance, axis=0)
    return synthetic_performance_list, flag



ANOMALY_PARAM_GRID = {
    'spikes': {
        'anomaly_type': ['spikes'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
        'anomaly_propensity': [0.5],
    },
    'contextual': {
        'anomaly_type': ['contextual'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
    },
    'flip': {
        'anomaly_type': ['flip'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
    },
    'speedup': {
        'anomaly_type': ['speedup'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        # 'speed': [0.25, 0.5, 2, 4],
        'speed': [0.25],
        'scale': [2],
    },
    'noise': {
        'anomaly_type': ['noise'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'noise_std': [0.05],
        'scale': [2],
    },
    'cutoff': {
        'anomaly_type': ['cutoff'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        # 'constant_type': ['noisy_0', 'noisy_1'],
        'constant_type': ['noisy_0'],
        'constant_quantile': [0.75],
        'scale': [2],
    },
    'scale': {
        'anomaly_type': ['scale'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        # 'amplitude_scaling': [0.25, 0.5, 2, 4],
        'amplitude_scaling': [0.25],
        'scale': [2],
    },
    'wander': {
        'anomaly_type': ['wander'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'baseline': [-0.3, -0.1, 0.1, 0.3],
        'scale': [2],
    },
    'average': {
        'anomaly_type': ['average'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'ma_window': [4, 8],
        'scale': [2],
    }
}