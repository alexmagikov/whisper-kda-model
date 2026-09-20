from __future__ import annotations
import torch
import pandas as pd
import numpy as np
from transformers import (
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)
from datasets import load_from_disk
import evaluate

class WhisperBaseline:
    def __init__(
        self,
        model_name="openai/whisper-tiny",
        audio_column="audio",
        text_column="normalized_sentence",
        language="russian",
        task="transcribe",
        device=None,
    ):
        self.model_name = model_name
        self.audio_column = audio_column
        self.text_column = text_column
        self.language = language
        self.task = task
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self._load_model()
        self.wer_metric = evaluate.load("wer")
        self.results = None
        self.dataset_wer = None

    def _load_model(self):
        self.feature_extractor = (
            WhisperFeatureExtractor.from_pretrained(
                self.model_name
            )
        )

        self.tokenizer = (
            WhisperTokenizer.from_pretrained(
                self.model_name,
                language=self.language,
                task=self.task,
            )
        )

        self.model = (
            WhisperForConditionalGeneration
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(self.device)
        self.model.eval()
        self.forced_decoder_ids = (
            self.tokenizer.get_decoder_prompt_ids(
                language=self.language,
                task=self.task,
            )
        )

        print("Model loaded.")

    def prepare_audio(self, audio):
        waveform = audio["array"]
        sampling_rate = audio["sampling_rate"]
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs.input_features.to(
            self.device
        )

    @torch.no_grad()
    def transcribe(self, audio):
        input_features = self.prepare_audio(
            audio
        )
        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
        )

        prediction = self.tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return prediction.strip().lower()

    def predict_example(self, example,):
        prediction = self.transcribe(
            example[self.audio_column]
        )
        return {
            "reference": example[self.text_column],
            "prediction": prediction,
        }

    def predict_dataset(self, dataset, max_examples=None):
        results = []
        if max_examples is None:
            max_examples = len(dataset)
        for example in dataset.select(range(max_examples)):
            results.append(
                self.predict_example(example)
            )

        return results

    def compute_wer(self, reference: str, prediction: str) -> float:
        return self.wer_metric.compute(
            references=[reference],
            predictions=[prediction],
        )

    def evaluate(self, dataset, max_examples=None):
        predictions = self.predict_dataset(
            dataset,
            max_examples=max_examples,
        )
        rows = []
        references = []
        predicted = []
        for item in predictions:
            reference = item["reference"]
            prediction = item["prediction"]
            references.append(reference)
            predicted.append(prediction)
            rows.append({
                "reference": reference,
                "prediction": prediction,
                "wer": self.compute_wer(
                    reference,
                    prediction),}
            )

        self.results = pd.DataFrame(rows)
        self.dataset_wer = self.wer_metric.compute(
            references=references,
            predictions=predicted,
        )

        return self.results


    def average_wer(self):
        return self.results["wer"].mean()

    def save_csv(self, filename="baseline_results.csv",
    ):
        self.results.to_csv(
            filename,
            index=False,
            encoding="utf-8",
        )

    def print_report(self):
        print("WHISPER BASELINE REPORT")
        print(f"Examples : {len(self.results)}")
        print(f"Average sample WER: {self.average_wer():.4f}")
        print(f"Dataset WER: {self.dataset_wer:.4f}")

dataset = load_from_disk("preprocessed_dataset/whisper_prepocessed_dataset.arrow")
print(f"Загружено {len(dataset)} примеров")

whisper_validator = WhisperBaseline()

for split in dataset.keys():
    print(f"\nEvaluating {split}...")
    whisper_validator.evaluate(dataset[split])
    whisper_validator.print_report()
    whisper_validator.save_csv(f"{split}_results.csv")