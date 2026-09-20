from datasets import Dataset
from datasets import DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

class DatasetFeatureExtractor:
    def __init__(self,
                 audio_column="audio",
                 text_column="normalized_sentence",
                 sampling_rate=16000,
                 max_target_length=448,
                 feature_extractor=None,
                 tokenizer=None,
                 ):
        self.audio_column = audio_column
        self.text_column = text_column
        self.sampling_rate = sampling_rate
        self.max_target_length = max_target_length
        self.feature_extractor = (feature_extractor
                                  or WhisperFeatureExtractor.from_pretrained(
                    "openai/whisper-tiny"))

        self.tokenizer = (tokenizer
                          or WhisperTokenizer.from_pretrained(
                    "openai/whisper-tiny",
                    language="russian",
                    task="transcribe"))

    def _iterate_splits(self, dataset):
        if isinstance(dataset, DatasetDict):
            return dataset.items()

        if isinstance(dataset, Dataset):
            return [("dataset", dataset)]

        raise TypeError(
            "Expected Dataset or DatasetDict."
        )

    def _prepare_example(self, example):
        audio = example[self.audio_column]
        input_features = self.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        labels = self.tokenizer(
            example[self.text_column],
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        return {
            "input_features": input_features["input_features"][0],
            "labels": labels, 
            "attention_mask": input_features["attention_mask"][0]
        }

    def apply(self, dataset: Dataset | DatasetDict, num_proc=None):
        prepared = {}
        for split_name, split in self._iterate_splits(dataset):
            prepared[split_name] = split.map(
                self._prepare_example,
                remove_columns=split.column_names,
                num_proc=num_proc,
                desc=f"Preparing {split_name}",
            )

        return DatasetDict(prepared)

# dataset_builder = DatasetFeatureExtractor(
#     feature_extractor=WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny"),
#     tokenizer=my_tokenizer,  # сейчас WhisperTokenizer, потом — токенизатор Маргариты
# )