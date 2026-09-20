import torch

class DataCollator:
    def __init__(
        self,
        feature_extractor, 
        tokenizer,
        decoder_start_token_id,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        input_features = [
            {"input_features": example["input_features"],
             "attention_mask": example["attention_mask"]}
            for example in features
        ]

        batch = self.feature_extractor.pad(
            input_features,
            return_attention_mask=True,
            return_tensors="pt",
        )

        label_features = [
            {"input_ids": example["labels"]}
            for example in features
        ]

        labels_batch = self.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"] != 1,
            -100,
        )
        
        if labels.size(1) == 0:
            raise ValueError("Получен батч с пустыми labels")

        starts_with_decoder_start = labels[:, 0].eq(
            self.decoder_start_token_id
        )

        if starts_with_decoder_start.any():
            if not starts_with_decoder_start.all():
                raise ValueError(
                    "Часть labels начинается с decoder_start_token_id, "
                    "а часть нет."
                )

            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch