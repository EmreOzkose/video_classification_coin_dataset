# https://pytorchvideo.org/docs/tutorial_classification
import pytorch_lightning
import torch.utils.data

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

from dataset import Dataset
from classification_dataset import ClassificationDataset


class CoinDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self,):
        super().__init__()
        
        target_label_list = [
            ["MakeSandwich", "CookOmelet", "MakePizza", "MakeYoutiao", "MakeBurger", "MakeFrenchFries"],
            ["AssembleBed", "AssembleSofa", "AssembleCabinet", "AssembleOfficeChair"],
        ]
        
        self.df_dataset_train, self.df_dataset_test = self.get_dataset(target_label_list=target_label_list)

        self._BATCH_SIZE = 4
        self._NUM_WORKERS = 0

    def train_dataloader(self):
        train_dataset = ClassificationDataset(df_dataset=self.df_dataset_train)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        val_dataset = ClassificationDataset(df_dataset=self.df_dataset_test)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def get_dataset(self, download=False, target_label_list=None):
        dataset = Dataset(
            coin_json_path = "/home/emre/workspace/medium_repos/video_classification_coin_dataset/coin_dataset/COIN.json",
            taxonomy_path = "/home/emre/workspace/medium_repos/video_classification_coin_dataset/coin_dataset/target_action_mapping.csv"
        )
        class_names = ["cooking", "decoration"]

        df_dataset = dataset.create_dataset(target_label_list)
        if download:
            df_dataset = dataset.download_dataset(df_dataset=df_dataset, save_folder="/home/emre/workspace/medium_repos/video_classification_coin_dataset/coin_subset", drop_none=True)
        df_dataset = dataset.add_download_local_paths(df_dataset=df_dataset, save_folder="/home/emre/workspace/medium_repos/video_classification_coin_dataset/coin_subset", drop_none=True)

        train_ratio = 0.8
        train_limit = int(len(df_dataset) * train_ratio)
        df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
        df_dataset = df_dataset[df_dataset["paths"] != "/home/emre/workspace/medium_repos/video_classification_coin_dataset/coin_subset/0/nKfAPLrOUF0.mp4"]

        df_dataset_train, df_dataset_test = df_dataset[:train_limit], df_dataset[train_limit:]
        return df_dataset_train, df_dataset_test


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.create_model()
        
        self.batch_size = 4
        self.train_ratio = 0.8
        
    def create_model(self, num_class=2, pretrained=True):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
        model.blocks[5].proj = nn.Linear(2048, num_class).to(self.device)
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self.model(inputs)
        loss = F.cross_entropy(y_hat, labels)
        _, preds = torch.max(y_hat, 1)
        accuracy = torch.sum(preds == labels.data).double() / inputs.shape[0]
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("accuracy", accuracy, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self.model(inputs)
        loss = F.cross_entropy(y_hat, labels)
        _, preds = torch.max(y_hat, 1)
        accuracy = torch.sum(preds == labels.data).double() / inputs.shape[0]
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("accuracy", accuracy, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
