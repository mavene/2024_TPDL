import torch
import json
import os
import numpy
import random

# Seeder for reproducibility
def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Saver / loader - model
def load_config(model_constructor, model_name, path='../models'):
    config_path = path + "/" + model_name + "/" + "config.json"
    with open(config_path, "r") as f:
        args = json.load(f)
        return model_constructor(**args)
    
def restore_model(model, model_name, epoch, path='../models'):
    model_path = path + "/" + model_name + "/" + f'{epoch}.pt'
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

def load_model(model_constructor, modelName, epoch, path='../models'):
    model = load_config(model_constructor, modelName, path)
    model = restore_model(model, modelName, epoch, path)