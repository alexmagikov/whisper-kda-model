import os
import json
import zipfile
from datasets import Dataset, Audio

def load_and_prepare_dataset(
        zip_path="subset_5h.zip",
        metadata_path="manifest_5h_subset.jsonl",
        audio_root="subset_5h",
        output_dir="prepared_data"):
    if not os.path.exists(audio_root):
        print(f"Распаковываю {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Архив распакован!")
    else:
        print(f"Папка {audio_root} уже существует, пропускаем распаковку.")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            filename = os.path.basename(sample["audio_filepath"])
            full_path = os.path.join(audio_root, filename)

            if not os.path.exists(full_path):
                print(f"Файл не найден: {full_path}")
                continue

            records.append({
                "audio": full_path,
                "sentence": sample["text"]
            })

    print(f"Собрано {len(records)} записей")

    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "whisper_dataset.arrow")
    dataset.save_to_disk(save_path)
    print(f"Датасет сохранен в: {save_path}")

    return dataset

if __name__ == "__main__":
    ds = load_and_prepare_dataset()
    print(ds[0]["sentence"])