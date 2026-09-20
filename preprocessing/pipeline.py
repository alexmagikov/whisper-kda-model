# основной пайплайн
from preprocessing.text_normalizer import TextNormalizer
from preprocessing.audio_preprocessor import AudioPreprocessor
from preprocessing.dataset_validator import DatasetValidator
from preprocessing.duplicate_remover import DuplicateRemover
from preprocessing.dataset_statistics import DatasetStatistics
from preprocessing.dataset_splitter import DatasetSplitter
from datasets import load_from_disk
import os

class Pipeline:
    def __init__(
        self,
        text_normalizer=None,
        audio_preprocessor=None,
        validator=None,
        duplicate_remover=None,
        statistics=None,
    ):
        self.text_normalizer = text_normalizer or TextNormalizer(lowercase=True, replace_yo=True,)
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor(sampling_rate=16000)
        self.validator = validator or DatasetValidator(min_duration=1, max_duration=30)
        self.duplicate_remover = duplicate_remover or DuplicateRemover(text_column="normalized_sentence", remove_text_duplicates=False, remove_audio_duplicates=True)
        self.statistics = statistics or DatasetStatistics()

    def run(self, dataset):
        print("Text normalization")
        dataset = self.text_normalizer.apply(dataset)
        print("Audio preprocessing")
        dataset = self.audio_preprocessor.apply(dataset)
        print("Validation")
        dataset = self.validator.validate(dataset)
        self.validator.report(dataset)
        dataset = self.validator.clean(dataset)
        print("Duplicate removal")
        dataset = self.duplicate_remover.apply(dataset)
        self.duplicate_remover.report()
        print("Dataset statistics")
        self.statistics.compute(dataset)
        self.statistics.print_report()

        return dataset


dataset = load_from_disk("prepared_data/whisper_dataset.arrow")
print(f"Загружено {len(dataset)} примеров")

pipeline = Pipeline()
dataset = pipeline.run(dataset)

splitter = DatasetSplitter(
    train_size=0.8,
    validation_size=0.1,
    test_size=0.1,
)

dataset = splitter.apply(dataset)
print(dataset)

output_dataset_dir = "preprocessed_dataset"

os.makedirs(output_dataset_dir, exist_ok=True)
save_path = os.path.join(output_dataset_dir, "whisper_prepocessed_dataset.arrow")
dataset.save_to_disk(save_path)
print(f"Датасет сохранен в: {save_path}")