import numpy as np
from datasets import Dataset

class DatasetStatistics:
    def __init__(
            self,
            audio_column="audio",
            text_column="normalized_sentence",
    ):
        self.audio_column = audio_column
        self.text_column = text_column
        self.report = {}

    def _compute_split(self, dataset):
        durations = []
        text_lengths = []
        word_lengths = []
        sampling_rates = []
        for example in dataset:
            audio = example[self.audio_column]
            waveform = audio["array"]
            sr = audio["sampling_rate"]
            duration = len(waveform) / sr
            durations.append(duration)
            sampling_rates.append(sr)
            text = example[self.text_column]
            text_lengths.append(len(text))
            word_lengths.append(len(text.split()))
        return {
            "examples": len(dataset),
            "hours": sum(durations) / 3600,
            "duration_mean": float(np.mean(durations)),
            "duration_median": float(np.median(durations)),
            "duration_min": float(np.min(durations)),
            "duration_max": float(np.max(durations)),
            "characters_mean": float(np.mean(text_lengths)),
            "words_mean": float(np.mean(word_lengths)),
            "sampling_rates": sorted(set(sampling_rates))
        }

    def compute(self, dataset):
        self.report = self._compute_split(
            dataset
        )

        return self.report

    def print_report(self):
        print("DATASET STATISTICS")
        for key, value in self.report.items():
            print(f"{key:20s}: {value}")

    def save_json(self, filename):
        import json
        with open(filename, "w", encoding="utf8") as f:
            json.dump(
                self.report,
                f,
                indent=4,
                ensure_ascii=False,
            )