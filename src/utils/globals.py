import collections
import yaml
import os
import warnings

config = {}

def config_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = config_update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

def init_global_config(args):
    global config

    # load default config
    with open(args.default_config) as file:
        config = yaml.full_load(file)

    # load dataset config, overwrite parameters if double
    with open(config["data"]["dataset_config"]) as file:
        config_data_dependent = yaml.full_load(file)
    config = config_update(config, config_data_dependent)

    # load experiment config, overwrite parameters if double
    if args.experiment_folder != 'None':
        experiment_config = os.path.join(args.experiment_folder, 'exp_config.yaml')
        if os.path.exists(experiment_config):
            with open(experiment_config) as file:
                exp_config = yaml.full_load(file)
            config = config_update(config, exp_config)
        config['logging']['experiment_folder'] = args.experiment_folder
        config['logging']['run_name'] = os.path.basename(args.experiment_folder)
    else:
        warnings.warn("No experiment folder was given. Use ./output folder to store experiment results.")
        config['logging']['experiment_folder'] = './output'
        config['logging']['run_name'] = 'default'


