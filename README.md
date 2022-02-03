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