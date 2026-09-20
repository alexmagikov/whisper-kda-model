import re
from typing import Callable, List, Optional
from datasets import Dataset

class TextNormalizer:
    def __init__(
        self,
        text_column: str = "sentence",
        lowercase: bool = False,
        replace_yo: bool = True,
        remove_multiple_spaces: bool = True,
        remove_space_before_punctuation: bool = True,
        output_column: str = "normalized_sentence"
    ):
        self.text_column = text_column
        self.lowercase = lowercase
        self.replace_yo = replace_yo
        self.remove_multiple_spaces = remove_multiple_spaces
        self.remove_space_before_punctuation = (
            remove_space_before_punctuation
        )
        self.output_column = output_column

    def normalize_text(self, text: str) -> str:
        if text is None:
            return text

        if self.lowercase:
            text = text.lower()

        if self.replace_yo:
            text = text.replace("ё", "е")

        if self.remove_multiple_spaces:
            text = re.sub(r"\s+", " ", text)

        if self.remove_space_before_punctuation:
            text = re.sub(
                r"\s+([.,!?;:])",
                r"\1",
                text,
            )

        text = text.strip()

        return text

    def _map_function(self, example):
        example[self.output_column] = self.normalize_text(
            example[self.text_column]
        )

        return example

    def apply(self, dataset: Dataset, num_proc: int | None = None) -> Dataset:
        return dataset.map(
            self._map_function,
            num_proc=num_proc,
            desc="Normalizing text",
        )