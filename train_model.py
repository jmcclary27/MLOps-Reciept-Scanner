import json
import os
import pandas as pd
import torch
import mlflow
import mlflow.transformers
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load config
with open("config.json", "r") as f:
    config = json.load(f)
mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
mlflow.set_experiment("receipt-ocr-finetune")

# Load processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or 0
model.config.pad_token_id = processor.tokenizer.pad_token_id or 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Dataset
class ReceiptDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        text = str(self.data.iloc[idx]['text'])

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = processor.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).input_ids.squeeze()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return pixel_values, labels


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return pixel_values, labels


# Training function
def train_model(csv_path, epochs=100, batch_size=4, lr=5e-5, patience=10):
    df = pd.read_csv(csv_path)
    #train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    #val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df, test_df, val_df = df, df, df
    
    train_loader = DataLoader(ReceiptDataset(train_df), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ReceiptDataset(val_df), batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(ReceiptDataset(test_df), batch_size=batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    patience_counter = 0

    with mlflow.start_run():
        mlflow.log_params({
            "model": "microsoft/trocr-base-stage1",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr
        })

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for pixel_values, labels in train_loader:
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for pixel_values, labels in val_loader:
                    pixel_values, labels = pixel_values.to(device), labels.to(device)
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }, step=epoch + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping due to no improvement in validation loss.")
                    break

        # Optional: Save locally for reuse/debugging
        model.save_pretrained("finetuned_trocr")
        processor.save_pretrained("finetuned_trocr")

        # Log the model to MLflow (skip input_example to avoid hang)
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "image_processor": processor.image_processor,
                "tokenizer": processor.tokenizer
            },
            name="model",
            task="image-to-text"
        )

        # Evaluate on test set
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for pixel_values, labels in test_loader:
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                test_loss += outputs.loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
        mlflow.log_metric("test_loss", avg_test_loss)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with image paths and texts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    train_model(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
