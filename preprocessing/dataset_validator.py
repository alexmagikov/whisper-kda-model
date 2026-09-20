from collections import Counter
from datasets import Dataset

class DatasetValidator:
    def __init__(
        self,
        min_duration=1.0,
        max_duration=30.0,
        min_text_length=1,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_text_length = min_text_length

    def _validate_example(self, example):
        example["is_valid"] = False
        example["error"] = None
        example["duration"] = None

        text = example.get("sentence")

        if text is None:
            example["error"] = "text_none"
            return example

        if not isinstance(text, str):
            example["error"] = "text_not_string"
            return example

        text = text.strip()

        if len(text) < self.min_text_length:
            example["error"] = "text_empty"
            return example

        try:
            audio = example["audio"]
            waveform = audio["array"]
            sr = audio["sampling_rate"]

        except Exception:
            example["error"] = "audio_decode_error"
            return example

        duration = len(waveform) / sr
        example["duration"] = duration

        if duration < self.min_duration:
            example["error"] = "too_short"
            return example

        if duration > self.max_duration:
            example["error"] = "too_long"
            return example

        example["is_valid"] = True
        return example

    def validate(self, dataset: Dataset):
        validated = dataset.map(
            self._validate_example,
            desc="Validating dataset"
        )

        return validated

    def report(self, dataset: Dataset):
        total = len(dataset)
        valid = 0
        errors = Counter()
        for example in dataset:
            if example["is_valid"]:
                valid += 1
            else:
                errors[example["error"]] += 1

        print("VALIDATION REPORT")
        print(f"Examples : {total}")
        print(f"Valid    : {valid}")
        print(f"Invalid  : {total-valid}")
        if errors:
            print("ERROR SUMMARY")
            for error, count in errors.items():
                print(
                    f"{error:25s} {count}"
                )

    def clean(self, dataset: Dataset):
        cleaned = dataset.filter(
            lambda x: x["is_valid"],
            desc="Removing invalid samples"
        )

        cleaned = cleaned.remove_columns(
            ["is_valid", "error"]
        )

        return cleaned