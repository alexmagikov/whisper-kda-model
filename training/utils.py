import torch

def check_generation_cache(
        model,
        prepared_dataset,
        data_collator,
        device,
    ):
        validation_example = prepared_dataset["validation"][0]

        batch = data_collator(
            [validation_example]
        )

        input_features = batch["input_features"].to(device)

        attention_mask = batch.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        model.eval()

        without_cache = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=16,
            use_cache=False,
        )

        with_cache = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=16,
            use_cache=True,
        )

        print("Without cache:", without_cache)
        print("With cache:   ", with_cache)

        if not torch.equal(without_cache, with_cache):
            raise RuntimeError(
                "Генерация с cache отличается от генерации "
                "без cache. Нельзя включать cache для validation."
            )

        print("KDA generation cache: OK")