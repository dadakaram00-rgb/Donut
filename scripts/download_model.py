import os
from transformers import VisionEncoderDecoderModel, DonutProcessor

def download_model():
    model_id = "naver-clova-ix/donut-base"
    output_dir = "models/donut-base"

    print(f"Downloading model {model_id} to {output_dir}...")

    # Download and save model
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.save_pretrained(output_dir)

    # Download and save processor
    processor = DonutProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)

    print("Download complete.")

if __name__ == "__main__":
    download_model()
