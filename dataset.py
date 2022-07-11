import os
import json
import yt_dlp
import pandas as pd

from tqdm import tqdm


class Taxonomy():
    def __init__(self, taxonomy_path) -> None:
        self.taxonomy_path = taxonomy_path
        df_taxonomy = pd.read_csv(taxonomy_path)

        self.target_id2label = {row["Target Id"]: row["Target Label"] for _, row in df_taxonomy.iterrows()}
        self.target_label2id = {j:i for i, j in self.target_id2label.items()}
        
        self.action_id2label = {row["Action Id"]: row["Action Label"] for _, row in df_taxonomy.iterrows()}


class Dataset():
    def __init__(self, coin_json_path, taxonomy_path) -> None:
        self.coin_json_path = coin_json_path
        self.taxonomy_path = taxonomy_path

        self.taxonomy = Taxonomy(taxonomy_path=taxonomy_path)
        self.raw_data = self.load_coin_json_from_file(coin_json_path)

    def load_coin_json_from_file(self, file_path):
        f = open(file_path)
        data = json.load(f)["database"]
        return data

    def create_dataset(self, target_label_list):
        target_ids = {i: [self.taxonomy.target_label2id[j] for j in target_label_list[i]] for i in range(len(target_label_list))}
        
        target_ids_reverse = {}
        for upper, target_id_list in target_ids.items():
            for each in target_id_list:
                target_ids_reverse[each] = upper
        
        dataset_list = []
        for sample_id, sample in self.raw_data.items():
            annotations = sample["annotation"]
            recipe_id = sample["recipe_type"]
            video_url = sample["video_url"]

            if recipe_id in target_ids_reverse.keys():
                for ann in annotations:
                    segment = f"{ann['segment'][0]}_{ann['segment'][1]}"
                    label = ann["label"]

                    dataset_list.append([target_ids_reverse[recipe_id], recipe_id, video_url, segment, label])

        pd_limited_data = pd.DataFrame(dataset_list, columns=["label", "action id", "url", "segment", "action label"])
        self.classes = pd_limited_data["label"].unique()
        
        return pd_limited_data

    def add_download_local_paths(self, df_dataset: pd.DataFrame, save_folder, drop_none) -> pd.DataFrame:
        paths = []

        os.makedirs(save_folder, exist_ok=True)
        for i in self.classes:
            os.makedirs(os.path.join(save_folder, str(i)), exist_ok=True)

        for index, sample in tqdm(df_dataset.iterrows(), desc="downloading COIN subset", total=len(df_dataset)):
            label = sample.label
            url = sample.url
            video_url_id = url.split("/")[-1]

            URLS = [url]

            save_path = os.path.join(save_folder, str(label), video_url_id) + ".mp4"
            if os.path.exists(save_path):
                paths.append(save_path)
            else:
                paths.append(None)

        df_dataset["paths"] = paths
        if drop_none:
            df_dataset = df_dataset.dropna()
            df_dataset = df_dataset.reset_index(drop=True)
        return df_dataset

    def download_dataset(self, df_dataset: pd.DataFrame, save_folder, drop_none) -> pd.DataFrame:
        paths = []

        os.makedirs(save_folder, exist_ok=True)
        for i in self.classes:
            os.makedirs(os.path.join(save_folder, str(i)), exist_ok=True)

        for index, sample in df_dataset.iterrows():
            label = sample.label
            url = sample.url
            video_url_id = url.split("/")[-1]

            URLS = [url]

            save_path = os.path.join(save_folder, str(label), video_url_id) + ".mp4"
            if os.path.exists(save_path): continue
            ydl_opts = {
                'format': 'mp4',
                'outtmpl': save_path
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    error_code = ydl.download(URLS)
                paths.append(save_path)
            except:
                paths.append(None)

        df_dataset["paths"] = paths

        if drop_none:
            df_dataset = df_dataset.dropna()
            df_dataset = df_dataset.reset_index(drop=True)
        return df_dataset
