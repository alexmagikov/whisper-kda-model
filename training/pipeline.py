from training.metrics import Metrics
from model.model import WhisperKDAModel, create_kda_config
from training.trainer import Trainer
from training.data_collactor import DataCollator
from training.dataset_feature_extractor import DatasetFeatureExtractor

from lion_pytorch import Lion
from datasets import load_from_disk
from transformers import WhisperFeatureExtractor, WhisperTokenizer, PreTrainedTokenizerFast
import torch
from training.utils import check_generation_cache
from preprocessing.text_normalizer import TextNormalizer

WHISPER_MODEL_NAME = "openai/whisper-tiny"
DATASET_PATH = "preprocessed_dataset/whisper_prepocessed_dataset.arrow"

WHISPER_MODEL_NAME = "openai/whisper-tiny"
DATASET_PATH = "preprocessed_dataset/whisper_prepocessed_dataset.arrow"

CHECKPOINT_PATH = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    WHISPER_MODEL_NAME,
    return_attention_mask=True
)

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    CHECKPOINT_PATH
) 

dataset = load_from_disk(DATASET_PATH)
print(f"Загружено {len(dataset)} примеров")

dataset_builder = DatasetFeatureExtractor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

prepared_dataset = dataset_builder.apply(dataset, num_proc=None)

kda_config = create_kda_config(tokenizer=tokenizer)

model = WhisperKDAModel(
    config=kda_config
).to(device)

# использовать предобученный декодер
# model.load_pretrained_decoder(
#   CHECKPOINT_PATH
# )
# model = model.to(device)

trainable_params = [
    param
    for param in model.parameters()
    if param.requires_grad
]

optimizer = Lion(
    trainable_params,
    lr=3e-5,
    betas=(0.9, 0.99),
    weight_decay=0.1,
)

text_normalizer = TextNormalizer(
    lowercase=True,
    replace_yo=True,
)

metrics = Metrics(
    tokenizer=tokenizer,
    text_normalizer=text_normalizer
)

data_collator = DataCollator(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    decoder_start_token_id=kda_config.decoder_start_token_id,
)

check_generation_cache(
    model=model,
    prepared_dataset=prepared_dataset,
    data_collator=data_collator,
    device=device,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataset=prepared_dataset,
    data_collator=data_collator,
    metrics=metrics,
    tokenizer=tokenizer,
)

trainer.train()
