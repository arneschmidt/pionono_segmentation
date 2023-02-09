import argparse
import os
import mlflow
import pandas as pd

parser = argparse.ArgumentParser(description="Calculate mean and s.e. of cross validation.")
parser.add_argument("--exp_dir", "-e", type=str,
                    default="../../experiments/gleason19/experiments/cross_validation/pionono/",
                    help="Input directory of dataset.")
args = parser.parse_args()

exp_dir = args.exp_dir
cv_folders = ["cval0/", "cval1/", "cval2/", "cval3/"]
results_path = 'test/quantitative_results/results.csv'
output_file = 'crossval_results.csv'

df = pd.DataFrame()

failed_cvs = []

for cv in range(len(cv_folders)):
    path_input_results = exp_dir + cv_folders[cv] + results_path
    if os.path.isfile(path_input_results):
        cv_df = pd.read_csv(path_input_results)
        df = pd.concat([df, cv_df], axis=1)
    else:
        failed_cvs.append(cv_folders[cv])

mean = df.mean(axis='rows')
se = df.sem(axis='rows')

# todo: save to mlflow
for c in mean.columns():
    mlflow.log_metric(c, mean[c])
for c in se.columns():
    mlflow.log_metric(c, se[c])

if len(failed_cvs) > 0:
    mlflow.log_param('failed_cvs', failed_cvs)