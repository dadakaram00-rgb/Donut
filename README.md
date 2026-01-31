# Donut Fine-tuning on German Invoices (CPU)

This repository contains scripts to fine-tune the Donut (Document Understanding Transformer) model on German scanned PDF-like images (invoices with tables). The code is optimized to run on CPU.

## 1. Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 2. Project Structure

```
.
├── data/               # Stores images and metadata.jsonl
├── models/             # Stores downloaded and fine-tuned models
├── scripts/
│   ├── download_model.py  # Download base model (run online)
│   ├── generate_data.py   # Generate synthetic training data
│   ├── train.py           # Fine-tune the model (CPU compatible)
│   └── inference.py       # Run inference on an image
├── requirements.txt
└── README.md
```

## 3. Data Preparation

The training data consists of images (JPG/PNG) and a `metadata.jsonl` file.

### Metadata Format

The `metadata.jsonl` file contains one line per image:

```json
{"file_name": "invoice_0.jpg", "ground_truth": "{\"gt_parse\": {\"company\": \"...\", \"date\": \"...\", \"items\": [...], \"total\": \"...\"}}"}
```

- `file_name`: Name of the image file in the `data/` directory.
- `ground_truth`: A JSON string containing the `gt_parse` key, which maps to the actual JSON structure you want the model to learn.

### Generating Synthetic Data

You can generate sample German invoice data using the provided script:

```bash
python3 scripts/generate_data.py
```

This will populate the `data/` directory with 10 synthetic invoice images and the `metadata.jsonl` file.

## 4. Downloading the Model (Online Step)

If you are working offline, you first need to download the base model while you have internet access (e.g., on Google Colab).

```bash
python3 scripts/download_model.py
```

This will download `naver-clova-ix/donut-base` to `models/donut-base`.

**For Colab Users:**
After running the download script, zip the `models` folder and download it to your local machine.

```bash
zip -r models.zip models/
```

Then upload `models.zip` to your offline PC and unzip it in the project root.

## 5. Training (CPU)

To fine-tune the model on your CPU:

```bash
python3 scripts/train.py
```

This script:
- Loads the base model from `models/donut-base`.
- Loads the dataset from `data/`.
- Fine-tunes the model (default: 3 epochs).
- Saves the fine-tuned model to `models/donut-finetuned`.

**Note:** Training on CPU is slow. For a real dataset, expect it to take a significant amount of time. The provided script uses a small batch size to accommodate CPU memory.

## 6. Inference

To test the model on an image:

```bash
python3 scripts/inference.py --image data/invoice_0.jpg
```

This will output the generated JSON structure extracted from the image.

## 7. Data and Table Format

The model learns to map the image pixels directly to the JSON text.
For tables (like line items in an invoice), ensure your JSON structure (in `gt_parse`) represents them logically, for example:

```json
{
  "items": [
    {"desc": "Item 1", "qty": "1", "price": "10.00"},
    {"desc": "Item 2", "qty": "2", "price": "20.00"}
  ]
}
```

The model will learn to generate this structure.
