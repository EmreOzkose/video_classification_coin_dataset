import torch
import torch.nn as nn

import json
from pytorchvideo.data.encoded_video import EncodedVideo

import torchvision
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
# https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df_dataset) -> None:
        super().__init__()

        self.df_dataset = df_dataset

        side_size = 200
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 200
        num_frames = 8
        sampling_rate = 8
        frames_per_second = 30

        self.transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        self.clip_duration = (num_frames * sampling_rate)/frames_per_second

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        video_path = self.df_dataset.iloc[idx]["paths"]
        start_time, end_time = self.df_dataset.iloc[idx]["segment"].split("_")
        start_time, end_time = float(start_time), float(end_time)
        label = self.df_dataset.iloc[idx]["label"]

        end_sec = start_time + self.clip_duration

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
        video_data = self.transform(video_data)
        inputs = video_data["video"]

        return inputs, label
