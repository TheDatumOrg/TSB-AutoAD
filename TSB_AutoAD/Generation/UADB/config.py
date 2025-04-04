import os
import argparse


class Config(object):
    def __init__(self):
        # ------- Basic Arguments -------
        # random seed
        self.seed = 0  
        self.log_file = 'none'
        # please set 'CUDA_VISIBLE_DEVICES' when calling python
        self.device = 'cuda:0'

        # ------- Data Arguments -------
        # select one of the 84 tabular datasets
        self.data_path = '../UADB/datasets/Yahoo_A1real_1_data.out'  
        self.realistic_synthetic_mode = 'none'
        self.noise_type = 'none'
        self.noise_ratio = 0.0
        self.duplicate_times = 1
        # select one of the 14 unsupervised anomaly detection models
        self.pseudo_model = 'lof'

        # ------- Optimization Arguments -------
        self.max_epochs = 10
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.sampler_per_epoch = 10000

        # ------- Model Abasements -------
        self.model = 'mlp'
        self.hidden_dim = 128
        self.output_dim = 32
        self.num_models = 1
        
        # ------- Experiments Settings -------
        # uadb base_mean base_std base_mean_cascade base_std_cascade
        self.experiment_type = 'uadb'
        self.initial_labels = None
        
    def update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    def to_dict(self):
        return dict(self.__dict__)


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def add_config_to_argparse(config, arg_parser):
    """The helper for adding configuration attributes to the argument parser"""
    for key, val in config.to_dict().items():
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        elif isinstance(val, (int, float, str)):
            arg_parser.add_argument('--' + key, type=type(val), default=val)
        else:
            raise Exception('Do not support value ({}) type ({})'.format(val, type(val)))


def get_config_from_command():
    # add arguments to parser
    config = Config()
    parser = argparse.ArgumentParser(description='Wind Power Forecasting')
    add_config_to_argparse(config, parser)

    # parse arguments from command line
    args = parser.parse_args()
    # update config by args
    config.update_by_dict(args.__dict__)

    return config
