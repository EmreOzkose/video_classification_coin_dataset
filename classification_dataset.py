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

        # video = EncodedVideo.from_path(video_path)

        end_sec = start_time + self.clip_duration

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
        video_data = self.transform(video_data)
        inputs = video_data["video"]
        # print("in ", start_time, end_time, inputs.shape)

        return inputs, label

    def read_video(video_path):
        stream = "video"
        video = torchvision.io.VideoReader(video_path, stream)
        video.get_metadata()

        video.set_current_stream("audio")

        frames = []
        for frame in video:
            frames.append(frame['data'])
        print(len(frames))
        frames = torch.stack(frames)
        print(frames.shape)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()
        self.alpha = 4
        self.device = "cuda:0"

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway.to(self.device), fast_pathway.to(self.device)]
        return frame_list


class ClassificationDataset2(torch.utils.data.Dataset):
    def __init__(self, df_dataset) -> None:
        super().__init__()

        self.df_dataset = df_dataset

        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        sampling_rate = 2
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
                    CenterCropVideo(crop_size),
                    PackPathway()
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

        video = EncodedVideo.from_path(video_path)

        end_sec = start_time + self.clip_duration

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
        video_data = self.transform(video_data)
        inputs = video_data["video"]
        del video_data
        # print("in ", start_time, end_time, inputs.shape)

        return inputs, label
