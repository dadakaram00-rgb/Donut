import os
import argparse
import json
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor

# Strict Offline Mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def inference(image_path, model_path="models/donut-finetuned"):
    print(f"Loading model from {model_path}...")

    # Strict Local Check
    if not os.path.exists(model_path):
        # Fallback to local base model if finetuned doesn't exist
        print(f"Model path {model_path} does not exist. Checking local base model at models/donut-base...")
        model_path = "models/donut-base"
        if not os.path.exists(model_path):
             raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Since you are offline, you must first run 'python scripts/download_model.py' "
                "on an online machine and transfer the 'models' directory to this machine."
            )

    processor = DonutProcessor.from_pretrained(model_path, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)

    # Move to CPU explicitly (though it is default)
    device = "cpu"
    model.to(device)
    model.eval()

    print(f"Processing image {image_path}...")
    image = Image.open(image_path).convert("RGB")

    # Resize image to match training if necessary (optional but good for consistency)
    # processor.image_processor.size = {"height": 1280, "width": 960}

    # Prepare input
    # Removed random_padding argument to avoid warnings
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate
    outputs = model.generate(
        pixel_values,
        max_length=512, # Matching training config
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # Remove cls token if present (it acts as start token)
    sequence = sequence.replace(processor.tokenizer.cls_token, "")

    print("Generated Output:")
    print(sequence)

    # Try to parse as JSON
    try:
        # DonutProcessor has a token2json method, but it expects a specific format (CORD-like)
        # Since we trained on raw JSON strings, we might just try json.loads directly
        # But let's see if token2json works or if we need custom parsing.
        # For this script, we'll try json.loads first.
        json_output = json.loads(sequence)
        print("\nParsed JSON:")
        print(json.dumps(json_output, indent=2))
    except Exception as e:
        print(f"\nCould not parse as JSON directly: {e}")
        # Fallback to token2json if it's in a format it understands
        try:
            json_output = processor.token2json(sequence)
            print("\nParsed JSON (via token2json):")
            print(json.dumps(json_output, indent=2))
        except Exception as e2:
            print(f"Could not parse via token2json: {e2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/donut-finetuned", help="Path to model")
    args = parser.parse_args()

    inference(args.image, args.model)
