import hashlib
from datasets import Dataset

class DuplicateRemover:
    def __init__(
            self,
            text_column="normalized_sentence",
            audio_column="audio",
            remove_text_duplicates=False,
            remove_audio_duplicates=True,
    ):
        self.text_column = text_column
        self.audio_column = audio_column
        self.remove_text_duplicates = remove_text_duplicates
        self.remove_audio_duplicates = remove_audio_duplicates
        self.removed_text = 0
        self.removed_audio = 0

    def _audio_hash(self, audio):
        array = audio["array"]

        return hashlib.sha256(
            array.tobytes()
        ).hexdigest()

    def _remove_text_duplicates(self, dataset):
        seen = set()
        before = len(dataset)

        def keep(example):
            text = example[self.text_column]
            if text in seen:
                return False
            seen.add(text)
            return True

        dataset = dataset.filter(
            keep,
            desc="Removing text duplicates"
        )
        removed = before - len(dataset)
        return dataset, removed

    def _remove_audio_duplicates(self, dataset):
        seen = set()
        before = len(dataset)

        def keep(example):
            audio_hash = self._audio_hash(
                example[self.audio_column]
            )
            if audio_hash in seen:
                return False
            seen.add(audio_hash)
            return True

        dataset = dataset.filter(
            keep,
            desc="Removing audio duplicates"
        )
        removed = before - len(dataset)
        return dataset, removed

    def apply(self, dataset):
        cleaned_splits = {}
        self.removed_text = 0
        self.removed_audio = 0
        if self.remove_text_duplicates:
            dataset, self.removed_text = (
                self._remove_text_duplicates(
                    dataset
                )
            )
        if self.remove_audio_duplicates:
            dataset, self.removed_audio = (
                self._remove_audio_duplicates(
                    dataset
                )
            )

        return dataset

    def report(self):
        print("DUPLICATES REPORT")
        if self.remove_text_duplicates:
            print(f"Text duplicates removed: {self.removed_text}")
        if self.remove_audio_duplicates:
            print(f"Audio duplicates removed: {self.removed_audio}")