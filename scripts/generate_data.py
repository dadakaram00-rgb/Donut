import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

# Create data directory
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metadata file
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")

def generate_invoice(idx):
    width, height = 800, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Content
    company_names = ["GmbH & Co. KG", "Müller GmbH", "Schmidt AG", "Auto Werkstatt"]
    company = random.choice(company_names)

    date = f"{random.randint(1, 28)}.{random.randint(1, 12)}.2023"
    total_amount = 0.0

    items = []
    for i in range(random.randint(1, 5)):
        price = random.uniform(10, 100)
        qty = random.randint(1, 10)
        item_total = price * qty
        total_amount += item_total

        item = {
            "desc": f"Artikel {i+1}",
            "qty": str(qty),
            "price": f"{price:.2f}"
        }
        items.append(item)

    total = f"{total_amount:.2f} EUR"

    # Draw text
    draw.text((50, 50), f"Rechnung von {company}", fill="black", font=font)
    draw.text((600, 50), f"Datum: {date}", fill="black", font=small_font)

    # Draw Table
    y = 150
    draw.text((50, y), "Beschreibung", fill="black", font=small_font)
    draw.text((400, y), "Menge", fill="black", font=small_font)
    draw.text((600, y), "Preis", fill="black", font=small_font)
    y += 30
    draw.line((50, y, 750, y), fill="black")
    y += 10

    for item in items:
        draw.text((50, y), item["desc"], fill="black", font=small_font)
        draw.text((400, y), item["qty"], fill="black", font=small_font)
        draw.text((600, y), item["price"], fill="black", font=small_font)
        y += 30

    draw.line((50, y, 750, y), fill="black")
    y += 20
    draw.text((600, y), f"Gesamt: {total}", fill="black", font=font)

    filename = f"invoice_{idx}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    # GT JSON
    # Structure: {"company": ..., "date": ..., "items": [...], "total": ...}
    gt_json = {
        "company": company,
        "date": date,
        "items": items,
        "total": total
    }

    entry = {
        "file_name": filename,
        "ground_truth": json.dumps({"gt_parse": gt_json})
    }

    return entry

def main():
    print("Generating synthetic data...")
    with open(METADATA_FILE, "w") as f:
        for i in range(10):
            entry = generate_invoice(i)
            f.write(json.dumps(entry) + "\n")
    print(f"Data generation complete. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
