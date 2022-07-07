import datetime
import pandas as pd


class Analyse():
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset

    def class_durations(self):
        classes = self.dataset["label"].unique()
        class_durations = {i: 0 for i in classes}

        for index, sample in self.dataset.iterrows():
            label = sample["label"]
            segment = sample["segment"]

            start, end = segment.split("_")
            start, end = float(start), float(end)

            class_durations[label] += end - start
        
        class_durations = {i: self.seconds2time(j) for i, j in class_durations.items()}
        print(class_durations)

    def seconds2time(self, n: int):
        return str(datetime.timedelta(seconds = n))


if __name__ == "__main__":
    pd_limited_data = ...
    analyse = Analyse(pd_limited_data)
    analyse.class_durations()
