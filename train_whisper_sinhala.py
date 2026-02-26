import json
import multiprocessing
import os

import torchaudio
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# ------------------
# Paths
# ------------------
raw_file_path = "C:/Projects/VideoTranscriber/SinhalaTrainingData/trainingData/metadata8.jsonl"
processed_dataset_dir = "C:/Projects/VideoTranscriber/SinhalaTrainingData/processed_dataset"


# ------------------
# Memory-efficient Dataset
# ------------------
class AudioTextDataset(Dataset):
    def __init__(self, jsonl_path, processor, processed_files=None):
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.processed_files = processed_files if processed_files else set()
        self.valid_offsets = []

        print("Scanning dataset file...")

        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
            offset = 0
            for line in f:
                line_bytes = line.encode("utf-8", errors="ignore")
                stripped = line.strip()

                if stripped:
                    try:
                        obj = json.loads(stripped)
                        file_path = obj.get("file")

                        if file_path and os.path.exists(file_path):
                            if file_path not in self.processed_files:
                                self.valid_offsets.append(offset)

                    except json.JSONDecodeError:
                        pass

                offset += len(line_bytes)

        print(f"Valid examples found: {len(self.valid_offsets)}")

    def __len__(self):
        return len(self.valid_offsets)

    def __getitem__(self, idx):
        print(f"Loading sample {idx}")
        while True:
            offset = self.valid_offsets[idx]

            with open(self.jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(offset)
                line = f.readline().strip()

            try:
                obj = json.loads(line)
                file_path = obj["file"]
                text = obj.get("text", "")
            except:
                idx = (idx + 1) % len(self.valid_offsets)
                continue

            try:
                waveform, sr = torchaudio.load(file_path)

                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

                inputs = self.processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=16000
                )

                with self.processor.as_target_processor():
                    labels = self.processor(text).input_ids

                return {
                    "input_features": inputs.input_features[0],
                    "labels": labels
                }

            except Exception:
                idx = (idx + 1) % len(self.valid_offsets)
                continue


# ------------------
# Pickle-safe Data Collator (IMPORTANT FOR WINDOWS)
# ------------------
class DataCollatorSpeechSeq2Seq:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = [x["input_features"] for x in batch]
        labels = [x["labels"] for x in batch]

        inputs = self.processor.feature_extractor(
            input_features,
            return_tensors="pt",
            padding=True
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                {"input_ids": labels},
                return_tensors="pt"
            )

        inputs["labels"] = labels_batch["input_ids"]
        return inputs


# ------------------
# MAIN FUNCTION
# ------------------
def main():
    multiprocessing.freeze_support()

    model_name = "openai/whisper-base"

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="si",
        task="transcribe",
        sampling_rate=16000
    )

    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="si",
        task="transcribe"
    )
    model.config.suppress_tokens = []
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Load processed files if available
    processed_files = set()
    if os.path.exists(processed_dataset_dir):
        try:
            import datasets
            ds = datasets.load_from_disk(processed_dataset_dir)
            processed_files = set(ds["file"])
            print(f"Already processed {len(processed_files)} files.")
        except:
            processed_files = set()

    print("Building dataset...")
    dataset = AudioTextDataset(
        raw_file_path,
        processor=processor,
        processed_files=processed_files
    )

    print(f"Dataset size to process: {len(dataset)} examples")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-sinhala",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-6,
        warmup_steps=3000,
        num_train_epochs=2,
        fp16=True,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="no",  # correct for your transformers version
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none",
        max_steps=10
    )

    data_collator = DataCollatorSpeechSeq2Seq(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    print("Starting training loop...")
    trainer.train()


# ------------------
# ENTRY POINT (CRITICAL FOR WINDOWS)
# ------------------
if __name__ == "__main__":
    main()
