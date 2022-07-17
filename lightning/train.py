import sys
sys.path.append("..")


import pytorch_lightning
from utils import seed_everything
from lightning.datamodule import VideoClassificationLightningModule, CoinDataModule


if __name__ == "__main__":
    seed_everything(42)

    classification_module = VideoClassificationLightningModule()
    data_module = CoinDataModule()
    trainer = pytorch_lightning.Trainer(gpus=[0])
    trainer.fit(classification_module, data_module)
