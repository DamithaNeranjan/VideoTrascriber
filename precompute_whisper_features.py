import json
import os
import torchaudio
from datasets import Dataset
from transformers import WhisperProcessor
from tqdm import tqdm

# ------------------
# Paths
# ------------------
raw_file_path = "C:/Projects/VideoTranscriber/SinhalaTrainingData/trainingData/metadata8.jsonl"
output_dir = "C:/Projects/VideoTranscriber/SinhalaTrainingData/whisper_features"

model_name = "openai/whisper-base"

print("Loading processor...")
processor = WhisperProcessor.from_pretrained(
    model_name,
    language="si",
    task="transcribe"
)

samples = []

print("Reading metadata...")
with open(raw_file_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line.strip())
            samples.append(obj)
        except:
            continue

print(f"Total samples: {len(samples)}")

processed_data = []

for sample in tqdm(samples):
    file_path = sample["file"]
    text = sample.get("text", "")

    if not os.path.exists(file_path):
        continue

    try:
        waveform, sr = torchaudio.load(file_path)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000
        )

        with processor.as_target_processor():
            labels = processor(text).input_ids

        processed_data.append({
            "input_features": inputs.input_features[0],
            "labels": labels
        })

    except Exception:
        continue

print("Creating HuggingFace dataset...")
dataset = Dataset.from_list(processed_data)

print("Saving dataset to disk...")
dataset.save_to_disk(output_dir)

print("Precomputation complete.")
