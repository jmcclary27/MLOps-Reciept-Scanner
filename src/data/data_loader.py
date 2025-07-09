import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ReceiptDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.data = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image"]
        text = self.data.iloc[idx]["text"]

        image = Image.open(image_path).convert("RGB")

        # Prepare inputs
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding="max_length", truncation=True)

        # Remove batch dimension
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs
