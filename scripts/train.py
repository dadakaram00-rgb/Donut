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
                padded_labels.append(torch.cat([label, torch.tensor(remainder, dtype=torch.long)]))
            else:
                padded_labels.append(torch.cat([torch.tensor(remainder, dtype=torch.long), label]))

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

    def transform_batch(examples):
        # examples is a dict of lists: {"image": [PIL.Image, ...], "ground_truth": [str, ...]}
        images = examples["image"]
        ground_truths = examples["ground_truth"]

        # Image processing
        try:
            # Processor handles batch of images
            pixel_values = processor(images, random_padding=True, return_tensors="pt").pixel_values
        except Exception as e:
            print(f"Error processing images: {e}")
            return {}

        # Text processing
        batch_input_ids = []
        for gt in ground_truths:
            try:
                gt_str = json.loads(gt)["gt_parse"]
                # Convert to string if it's a dict/list
                if not isinstance(gt_str, str):
                    target_sequence = json.dumps(gt_str)
                else:
                    target_sequence = gt_str
            except:
                target_sequence = ""

            target_sequence = target_sequence + processor.tokenizer.eos_token

            # Tokenize single example
            input_ids = processor.tokenizer(
                target_sequence,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze() # Squeeze to get 1D tensor

            batch_input_ids.append(input_ids)

        # Stack input_ids
        labels = torch.stack(batch_input_ids)

        # Create a clone for labels where padding is -100 (though collator handles it, doing it here is fine too if collator expects it)
        # But wait, our custom collator expects 'labels' to be a list of tensors or a tensor.
        # Since we use set_transform, the collator receives a list of the dictionaries returned by this function if we were iterating.
        # But set_transform works differently. It replaces the item access.
        # When trainer accesses dataset[i], it gets the result of transform(batch_of_size_1) if not batched, or...
        # Wait, set_transform(transform, output_all_columns=False)
        # If we just access dataset[i], transform is called on the fly.

        # Let's adjust to return a dict of tensors
        # Note: if set_transform is used, the trainer gets a dict of values.

        # Labels: set pad tokens to -100
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    print("Processing dataset (on-the-fly)...")
    # Use set_transform instead of map to avoid writing to disk
    dataset.set_transform(transform_batch)

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
        remove_unused_columns=False, # We need to keep columns for the transform to work on them? No, transform replaces them.
        # But remove_unused_columns=True tries to inspect the model signature and remove columns from the dataset.
        # Since we use set_transform, we must ensure that the dataset *yields* the right columns.
        # set_transform output overrides the columns.
        # So we should be fine. But set remove_unused_columns=False to be safe with custom transforms usually.
        # However, we only output pixel_values and labels.
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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
