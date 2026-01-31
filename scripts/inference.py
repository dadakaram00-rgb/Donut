import os
import argparse
import json
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor

def inference(image_path, model_path="models/donut-finetuned"):
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Using base model for demonstration.")
        model_path = "models/donut-base"
        if not os.path.exists(model_path):
             model_path = "naver-clova-ix/donut-base"

    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    # Move to CPU explicitly (though it is default)
    device = "cpu"
    model.to(device)
    model.eval()

    print(f"Processing image {image_path}...")
    image = Image.open(image_path).convert("RGB")

    # Prepare input
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate
    outputs = model.generate(
        pixel_values,
        max_length=768,
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
