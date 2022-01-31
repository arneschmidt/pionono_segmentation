from typing import Dict
import mlflow

import src.utils.globals

def start_logging():
    config = src.utils.globals.config
    mlflow.set_tracking_uri(config["logging"]["mlruns_folder"])
    experiment = mlflow.set_experiment(experiment_name=config["data"]["dataset_name"])
    mlflow.start_run(experiment_id=experiment.experiment_id, run_name='test') #config["logging"]["run_name"])
    mlflow.log_params(config['model'])
    mlflow.log_params(config['data'])

def log_results(results, mode, step=None):

    formatted_results = {}

    for key in results.keys():
        new_key = mode + '_' + key
        formatted_results[new_key] = results[key]

    mlflow.log_metrics(formatted_results, step=step)

