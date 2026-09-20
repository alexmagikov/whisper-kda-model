from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch
from training.metrics import Metrics
from model.model import WhisperKDAModel

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        dataset,
        data_collator,
        metrics,
        tokenizer,
        output_dir="checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["validation"]
        self.data_collator = data_collator
        self.metrics = metrics
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.trainer = None
        self.create_trainer()

    def create_trainer(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            num_train_epochs=20,
            predict_with_generate=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="wer", # "eval_loss"
            greater_is_better=False,
            generation_max_length=448,
            max_steps=2, # Для smoke-test
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            optimizers=(self.optimizer, None),
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.metrics.compute,
            processing_class=self.tokenizer,
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()

    def save_model(self):
        self.trainer.save_model()