import torch
import utils.globals as globals

def load_model():
    model_path = globals.config['model']['load_model']
    model = torch.load(model_path)
    return model