import os
import random
import torch
import numpy as np


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_model(save_folder, name, model):
        torch.save(model.state_dict(), os.path.join(save_folder, name+".pt"))


def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    print(f"model is loaded from : {load_path}")
    return model
