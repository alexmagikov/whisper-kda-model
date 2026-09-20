from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import GroupShuffleSplit

class DatasetSplitter:
    def __init__(
            self,
            train_size: float = 0.8,
            validation_size: float = 0.1,
            test_size: float = 0.1,
            seed: int = 42,
            group_column: str = "client_id",
    ):
        if abs(train_size + validation_size + test_size - 1.0) > 1e-6:
            raise ValueError(
                "train_size + validation_size + test_size must equal 1."
            )

        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.seed = seed
        self.group_column = group_column

    def apply(self, dataset: Dataset) -> DatasetDict:
        if self.group_column in dataset.column_names:
            print(f"Column '{self.group_column}' found. Using group split.")

            return self._group_split(dataset)
        print(f"Column '{self.group_column}' not found. Using random split.")

        return self._random_split(dataset)

    def _random_split(self, dataset: Dataset) -> DatasetDict:
        train_test = dataset.train_test_split(
            test_size=1.0 - self.train_size,
            seed=self.seed,
            shuffle=True,
        )
        train_dataset = train_test["train"]
        temp_dataset = train_test["test"]
        validation_ratio = (
                self.validation_size
                / (self.validation_size + self.test_size)
        )
        validation_test = temp_dataset.train_test_split(
            test_size=1.0 - validation_ratio,
            seed=self.seed,
            shuffle=True,
        )
        validation_dataset = validation_test["train"]
        test_dataset = validation_test["test"]
        return DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset})

    def _group_split(self, dataset: Dataset) -> DatasetDict:
        groups = dataset[self.group_column]
        indices = list(range(len(dataset)))
        splitter = GroupShuffleSplit(
            n_splits=1,
            train_size=self.train_size,
            random_state=self.seed,
        )

        train_idx, temp_idx = next(
            splitter.split(indices, groups=groups)
        )

        train_dataset = dataset.select(train_idx)
        temp_dataset = dataset.select(temp_idx)
        groups_temp = temp_dataset[self.group_column]
        validation_ratio = (
                self.validation_size
                / (self.validation_size + self.test_size)
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            train_size=validation_ratio,
            random_state=self.seed,
        )

        val_idx, test_idx = next(
            splitter.split(
                range(len(temp_dataset)),
                groups=groups_temp,
            )
        )

        validation_dataset = temp_dataset.select(val_idx)
        test_dataset = temp_dataset.select(test_idx)
        return DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset})