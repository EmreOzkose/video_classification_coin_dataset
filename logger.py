import os
import torch
import logging

from shutil import copy
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, exp_path) -> None:
        self.exp_path = exp_path
        self.script_folder = os.path.join(exp_path, "files")
        self.trained_model_folder = os.path.join(exp_path, "models")
        self.tb_folder = os.path.join(exp_path, "tensorboard_logs")

        os.makedirs(self.exp_path, exist_ok=True)
        os.makedirs(self.tb_folder, exist_ok=True)
        os.makedirs(self.script_folder, exist_ok=True)
        os.makedirs(self.trained_model_folder, exist_ok=True)
        self.copy_related_files()

        logging.basicConfig(filename=os.path.join(exp_path, 'training.log'),
                            format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def log(self, text):
        logging.info(text)

    def copy_related_files(self):
        copy("train.py", self.script_folder)
        copy("dataset.py", self.script_folder)
        copy("classification_dataset.py", self.script_folder)
