import json
import os
import pandas as pd
import torch
import mlflow
import mlflow.transformers
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load MLFLOW URI securely from config file
with open("config.json", "r") as f:
    config = json.load(f)

mlflow.set_tracking_uri(config["mlflow_tracking_uri"])

mlflow.set_experiment("receipt-ocr-finetune")

# Load processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Dataset class
class ReceiptDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        text = str(self.data.iloc[idx]['text'])

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = processor.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).input_ids.squeeze()

        labels[labels == processor.tokenizer.pad_token_id] = -100  # ignore loss on padding

        return pixel_values, labels


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return pixel_values, labels


# Training function
def train_model(csv_path, epochs=5, batch_size=4, lr=5e-5):
    dataset = ReceiptDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    with mlflow.start_run():
        mlflow.log_param("model", "microsoft/trocr-base-stage1")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                pixel_values, labels = batch
                pixel_values, labels = pixel_values.to(device), labels.to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")
            mlflow.log_metric("loss", avg_loss, step=epoch + 1)

        # Save model
        output_dir = "finetuned_trocr"
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

        # Log model to MLflow
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="model",
            input_example={"image": "<sample_image.jpg>"},
            task="image-to-text"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with image paths and texts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()

    train_model(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
