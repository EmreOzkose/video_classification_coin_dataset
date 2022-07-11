import os
import torch
import torch.nn as nn

from glob import glob
from tqdm import tqdm
from logger import Logger
from dataset import Dataset
from natsort import natsorted
from utils import seed_everything
from utils import save_model, load_model
from classification_dataset import ClassificationDataset


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        logger.log(f'Epoch {epoch}/{num_epochs - 1}\n---')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(dataloaders[phase], desc= f"epoch {str(epoch).zfill(3)} | phase {phase.ljust(5)}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == "val":
                save_model(logger.trained_model_folder, f"{epoch}_val_acc_{epoch_acc:.2f}", model)

    return model


def find_last_saved_model(model_dir):
    files = glob(os.path.join(model_dir, "*.pt"))
    files = natsorted(files)
    return files[0]


if __name__ == "__main__":
    seed_everything(42)
    continue_to_exp = False

    logger = Logger(exp_path="exps/exp1")

    dataset = Dataset(
        coin_json_path = "coin_dataset/COIN.json",
        taxonomy_path = "coin_dataset/target_action_mapping.csv"
    )

    target_label_list = [
        ["MakeSandwich", "CookOmelet", "MakePizza", "MakeYoutiao", "MakeBurger", "MakeFrenchFries"],
        ["AssembleBed", "AssembleSofa", "AssembleCabinet", "AssembleOfficeChair"],
    ]
    class_names = ["cooking", "decoration"]

    df_dataset = dataset.create_dataset(target_label_list)
    # df_dataset = dataset.download_dataset(df_dataset=df_dataset, save_folder="coin_subset", drop_none=True)
    df_dataset = dataset.add_download_local_paths(df_dataset=df_dataset, save_folder="coin_subset", drop_none=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.log(f"device: {device}")

    train_ratio = 0.8
    train_limit = int(len(df_dataset) * train_ratio)
    df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
    df_dataset = df_dataset[df_dataset["paths"] != "coin_subset/0/nKfAPLrOUF0.mp4"]

    df_dataset_train, df_dataset_test = df_dataset[:train_limit], df_dataset[train_limit:]

    classification_dataset_train = ClassificationDataset(df_dataset=df_dataset_train)
    classification_dataset_test = ClassificationDataset(df_dataset=df_dataset_test)

    batch_size = 4
    classification_dataloader_train = torch.utils.data.DataLoader(classification_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
    classification_dataloader_test = torch.utils.data.DataLoader(classification_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    dataloaders = {"train": classification_dataloader_train, "val": classification_dataloader_test}
    dataset_sizes = {"train": len(classification_dataset_train), "val": len(classification_dataset_test)}

    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).to(device)
    model.blocks[5].proj = nn.Linear(2048, 2).to(device)
    
    if continue_to_exp:
        last_saved_model_path = find_last_saved_model(logger.trained_model_folder)
        model = load_model(model, last_saved_model_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler)
