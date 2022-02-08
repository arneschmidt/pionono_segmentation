# Semantic Segmentation with Crowdsourcing

## Install Requirements
* Use Miniconda/Anaconda to install the requirements with `conda env create -f environment.yml`
* Activate the environment with `conda activate seg_crowd_env`
`

## Configuration
* To run the model with the dummy dataset, simply use python `src/main.py`
* For experiments there are three levels of configurations:
    1. The default config
    2. The dataset config
    3. The experiment config
* The configuration will be loaded in this order and parameters will be overwritten
* How to define config paths:
    1. The default config: By argument `-dc [path/to/config.yaml]`
    2. The dataset config: In the default config `data: dataset_config: [path/to/dataset_config.yaml]`
    3. The experiment config: By changing the experiment folder `-ef [path/to/directory]`. Here a file `exp_config.yaml` is expected.
* Example: `python src/main.py -dc ../../experiments/segmentation_tnbc/config.yaml -ef ../../experiments/segmentation_tnbc/linknet`
    
    
## Collaboration
To implement something new:
1. Checkout the main branch: `git checkout main`
2. Pull newest version: `git pull` (If there are changes that you can dicard, do a `git stash` to delete them)
3. Open a branch: `git checkout -b branch_name_that_describes_new_feature`
4. Add new code:
    * implement
    * add: `git add file_that_changed`
    * commit: `git commit -m 'small description of change'`
5. In github create a pull request (PR), ask collaborators if new implementation looks good
6. In github merge pull request to the main branch


## Logging with mlflow
* For the python library of mlflow checkout https://www.mlflow.org/docs/latest/python_api/mlflow.html#module-mlflow 
* The most important functions are mlflow.log_param, mlflow.log_metric and mlflow.log_artifact
* for port-forwarding run `ssh -L 5002:127.0.0.1:5002 [user]@cvision.ugr.es`





