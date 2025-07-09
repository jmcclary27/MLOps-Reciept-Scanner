import os
import json
import pandas as pd

# Folder where images live
img_folder = "data/raw/img"

# Folder where JSON files live
json_folder = "data/raw/key"

# Create lists to collect data
image_paths = []
texts = []

# Loop over images
for img_name in os.listdir(img_folder):
    if img_name.endswith(".jpg"):
        base_name = os.path.splitext(img_name)[0]
        json_path = os.path.join(json_folder, base_name + ".json")
        
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Build combined string
            company = data.get("company", "")
            address = data.get("address", "")
            date = data.get("date", "")
            total = data.get("total", "")

            combined_text = f"{company}\n{address}\n{date}\nTotal: {total}"

            # Save paths and text
            image_paths.append(os.path.join(img_folder, img_name))
            texts.append(combined_text)
        else:
            print(f"⚠️ Warning: No JSON found for {img_name}, skipping...")

# Create dataframe
df = pd.DataFrame({"image": image_paths, "text": texts})

# Save to CSV
df.to_csv("data/receipt_dataset.csv", index=False, encoding="utf-8-sig")
print("Dataset CSV created: data/receipt_dataset.csv")
