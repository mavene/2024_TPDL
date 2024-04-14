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


# Saves the model hyperparameters
def save_json(dict, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    import json
    config_path = path + "/" + "config.json"
    with open(config_path, "w") as f:
        json.dump(dict, f)

# Saves the model
def save_model(model, optimizer, epoch, path):
    if not os.path.exists(path):
        os.makedirs(path)

    model_path = os.path.join(path, f'{epoch}.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

# Loads the model hyperparameters
def load_config(model_constructor, modelName, path='../models'):
    import json
    config_path = path + "/" + modelName + "/" + "config.json"
    with open(config_path, "r") as f:
        args = json.load(f)
        return model_constructor(**args)

# Load the model weights
def restore_model(model, modelName, epoch, path="../models"):
    model_path = path + "/" + modelName + "/" + str(epoch) + ".pt"
    # print(model_path)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

# Load model with hyperparameters and weights
def load_model(model_constructor, modelName, epoch, path='../models'):
    model = load_config(model_constructor, modelName, path)
    model = restore_model(model, modelName, epoch, path)
    return model
