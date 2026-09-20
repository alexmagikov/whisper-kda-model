import numpy as np
from datasets import Audio
from datasets import Dataset
from datasets import DatasetDict

class AudioPreprocessor:
    def __init__(
        self,
        audio_column: str = "audio",
        sampling_rate: int = 16000,
    ):
        self.audio_column = audio_column
        self.sampling_rate = sampling_rate

    def apply(self, dataset: Dataset) -> Dataset:
        dataset = dataset.cast_column(
            self.audio_column,
            Audio(
                sampling_rate=self.sampling_rate
            ),
        )

        return dataset