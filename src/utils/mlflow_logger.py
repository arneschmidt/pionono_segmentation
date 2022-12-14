from typing import Dict

import matplotlib.pyplot as plt
import mlflow

import utils.globals

def start_logging():
    config = utils.globals.config
    mlflow.set_tracking_uri(config["logging"]["mlruns_folder"])

    data_config_log = config['data'].copy()
    data_config_log.pop('visualize_images') # drop this because it is often to long to be logged

    # experiment = mlflow.set_experiment(experiment_name=config["data"]["dataset_name"])
    mlflow.set_experiment(experiment_name=config["data"]["dataset_name"])
    # with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='test') as run:
    # mlflow.start_run(experiment_id=experiment.experiment_id, run_name='test')
    mlflow.start_run(run_name=config['logging']['run_name'])
    print('tracking uri:', mlflow.get_tracking_uri())
    print('artifact uri:', mlflow.get_artifact_uri())
    mlflow.log_params(config['model'])
    mlflow.log_params(data_config_log)
    mlflow.set_tags(config['logging']['tags'])
    mlflow.log_artifact('config.yaml')


def log_results(results, mode, step=None):

    formatted_results = {}

    for key in results.keys():
        new_key = mode + '_' + key
        formatted_results[new_key] = results[key]

    mlflow.log_metrics(formatted_results, step=step)


def probabilistic_model_logging(model, step):
    method = utils.globals.config['model']['method']
    loss_dict = {}
    if method == 'prob-unet':
        loss_dict['loss_reconstruction'] = float(model.reconstruction_loss.cpu().detach().numpy())
        loss_dict['loss_kl'] = float((model.kl * model.beta).cpu().detach().numpy())
        loss_dict['loss_regularization'] = float(model.reg_loss.cpu().detach().numpy())
    elif method == 'pionono':
        loss_dict['loss_log_likelihood'] = float(model.log_likelihood_loss.cpu().detach().numpy())
        loss_dict['loss_kl'] = float(model.kl_loss.cpu().detach().numpy())
        loss_dict['loss_regularization'] = float(model.reg_loss.cpu().detach().numpy())
    mlflow.log_metrics(loss_dict, step=step)




