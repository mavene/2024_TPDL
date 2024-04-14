import torch
import json
import os
import pandas
import numpy
import random

from torch.utils.data import Dataset

import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

# Seeder for reproducibility
def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# Implementation of Custom Dataset Class for CheXPhoto Dataset
class CheXDataset(Dataset):
    # Accepts dataframe object and str
    def __init__(self, df: pandas.DataFrame, px_size: int = 256):
        self.dataframe = df.copy()
        self.px_size = px_size
        self.transform = T.Compose([
            v2.Resize((self.px_size, self.px_size), interpolation=T.InterpolationMode.BICUBIC)
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x_path = DATA_PATH + "/" + self.dataframe.iloc[idx, 0].split("CheXphoto-v1.0", 1)[-1]
        resized_x_tensor = self.transform(read_image(x_path, mode = ImageReadMode.RGB)) /255
        y = torch.tensor(self.dataframe.iloc[idx, 1]).type(torch.LongTensor)
        return resized_x_tensor, y

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