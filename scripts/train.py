import os
import json
import torch
from transformers import VisionEncoderDecoderModel, DonutProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset

class DonutDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        pixel_values = [feature["pixel_values"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # Stack pixel values
        batch = {
            "pixel_values": torch.stack(pixel_values)
        }

        # Pad labels
        max_label_length = max(len(l) for l in labels)
        padding_side = self.processor.tokenizer.padding_side
        pad_token_id = self.processor.tokenizer.pad_token_id

        padded_labels = []
        for label in labels:
            remainder = [pad_token_id] * (max_label_length - len(label))
            if padding_side == "right":
                padded_labels.append(torch.cat([label, torch.tensor(remainder)]))
            else:
                padded_labels.append(torch.cat([torch.tensor(remainder), label]))

        batch["labels"] = torch.stack(padded_labels)

        # Set -100 for pad tokens in labels so they are ignored in loss
        batch["labels"][batch["labels"] == pad_token_id] = -100

        return batch

def train():
    # Configuration
    base_model_path = "models/donut-base"
    # If base model doesn't exist locally, fallback to huggingface hub
    if not os.path.exists(base_model_path):
        print(f"Local model not found at {base_model_path}, using naver-clova-ix/donut-base")
        base_model_path = "naver-clova-ix/donut-base"

    data_dir = "data"
    output_dir = "models/donut-finetuned"

    print("Loading model and processor...")
    # Load processor and model
    processor = DonutProcessor.from_pretrained(base_model_path)
    model = VisionEncoderDecoderModel.from_pretrained(base_model_path)

    # Configure model for training
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Prepare dataset
    print("Loading dataset...")
    # We use the 'imagefolder' builder which reads the metadata.jsonl automatically
    try:
        dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Constants
    max_length = 768
    image_size = processor.image_processor.size
    # Handle different size formats (some versions use shortest_edge, some height/width)
    if "height" in image_size:
        height, width = image_size["height"], image_size["width"]
    else:
        # Default fallback or logic for shortest_edge
        height, width = 2560, 1920

    def transform(example):
        # Image processing
        try:
            pixel_values = processor(example["image"], random_padding=True, return_tensors="pt").pixel_values
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

        # Text processing
        # The ground truth is in "ground_truth" column as a string `{"gt_parse": ...}`
        try:
            gt_str = json.loads(example["ground_truth"])["gt_parse"]
            # Convert to string if it's a dict/list
            if not isinstance(gt_str, str):
                target_sequence = json.dumps(gt_str)
            else:
                target_sequence = gt_str
        except:
            target_sequence = ""

        target_sequence = target_sequence + processor.tokenizer.eos_token

        # Tokenize
        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels.squeeze(),
        }

    print("Processing dataset...")
    processed_dataset = dataset.map(transform, remove_columns=["image", "ground_truth"])

    # IMPORTANT: Set format to torch so that the collator receives tensors, not lists
    processed_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # Custom Data Collator
    data_collator = DonutDataCollator(processor)

    # Training args
    # CPU optimization: use_cpu=True
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Small batch size for CPU
        gradient_accumulation_steps=4,
        num_train_epochs=3, # Small number for demo
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=100,
        eval_strategy="no",
        use_cpu=True, # Force CPU
        remove_unused_columns=True,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )

    print("Starting training on CPU...")
    trainer.train()

    print(f"Training complete. Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
