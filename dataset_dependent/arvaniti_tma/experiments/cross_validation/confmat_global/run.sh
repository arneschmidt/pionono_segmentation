python src/main.py -c ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/base_config.yaml -dc ../../experiments/arvaniti_tma/data_configs/data_config_crossval0.yaml -ef ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/cval0
python src/main.py -c ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/base_config.yaml -dc ../../experiments/arvaniti_tma/data_configs/data_config_crossval1.yaml -ef ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/cval1
python src/main.py -c ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/base_config.yaml -dc ../../experiments/arvaniti_tma/data_configs/data_config_crossval2.yaml -ef ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/cval2
python src/main.py -c ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/base_config.yaml -dc ../../experiments/arvaniti_tma/data_configs/data_config_crossval3.yaml -ef ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/cval3

python src/postprocessing_tools/calculate_results.py -e ../../experiments/arvaniti_tma/experiments/cross_validation/confmat_global/