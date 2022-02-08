import functools
import numpy as np

def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x


def get_preprocessing_params():
    formatted_settings = {}
    formatted_settings["input_range"] = (0,1)
    formatted_settings["input_space"] = "RGB"
    formatted_settings["mean"] = None
    formatted_settings["std"] = None
    return formatted_settings


def get_preprocessing_fn_without_normalization():
    params = get_preprocessing_params()
    return functools.partial(preprocess_input, **params)