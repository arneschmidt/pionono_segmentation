import argparse
import os
import mlflow
import pandas as pd

parser = argparse.ArgumentParser(description="Calculate mean and s.e. of cross validation.")
parser.add_argument("--exp_dir", "-e", type=str,
                    default="../../experiments/arvaniti_tma/experiments/cross_validation/punet/",
                    help="Input directory of dataset.")
args = parser.parse_args()

mlruns_folder = '/work/work_arne/mlflow_server'

exp_dir = args.exp_dir
cv_folders = ["cval0/", "cval1/", "cval2/", "cval3/"]
results_path = 'test/quantitative_results/results.csv'
output_file = 'crossval_results.csv'

df = pd.DataFrame()

failed_cvs = []

if not os.path.isdir(exp_dir):
    raise Exception('Directory not found!')

for cv in range(len(cv_folders)):
    path_input_results = exp_dir + cv_folders[cv] + results_path
    if os.path.isfile(path_input_results):
        cv_df = pd.read_csv(path_input_results, header=None, index_col=0)
        df = pd.concat([df, cv_df], axis=1)
    else:
        failed_cvs.append(cv_folders[cv])

mean = df.mean(axis='columns').rename('test_{}_mean'.format)
se = df.sem(axis='columns').rename('test_{}_se'.format)

mlflow.set_tracking_uri(mlruns_folder)

# derive dataset
if 'gleason19' in exp_dir:
    dataset_name = 'gleason19_crowdsourcing'
else:
    dataset_name = 'arvaniti_tma_crowdsourcing'
mlflow.set_experiment(experiment_name=dataset_name)

# derive run name
run_name = exp_dir.split("/")[-2] + '_crossval_results'
mlflow.start_run(run_name=run_name)

# todo: save to mlflow
mlflow.log_metrics(mean)
mlflow.log_metrics(se)
mlflow.set_tag('mode', 'results')

mlflow.log_param('failed_cvs', failed_cvs)
print('Failed cvs:')
print(failed_cvs)