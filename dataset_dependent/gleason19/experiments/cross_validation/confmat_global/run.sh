python src/main.py -c ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/base_config.yaml -dc ./dataset_dependent/gleason19/data_configs/data_config_crossval0.yaml -ef ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/cval0
python src/main.py -c ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/base_config.yaml -dc ./dataset_dependent/gleason19/data_configs/data_config_crossval1.yaml -ef ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/cval1
python src/main.py -c ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/base_config.yaml -dc ./dataset_dependent/gleason19/data_configs/data_config_crossval2.yaml -ef ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/cval2
python src/main.py -c ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/base_config.yaml -dc ./dataset_dependent/gleason19/data_configs/data_config_crossval3.yaml -ef ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/cval3

python src/postprocessing_tools/calculate_results.py -e ./dataset_dependent/gleason19/experiments/cross_validation/confmat_global/