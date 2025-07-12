# src/models/train_model.py

import os
import mlflow
import mlflow.pytorch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import torch
from PIL import Image

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MLFLOW_TRACKING_URI = "http://<YOUR_MLFLOW_VM_IP>:5000"
EXPERIMENT_NAME = "trocr_receipt_finetuning"
MODEL_NAME = "microsoft/trocr-base-handwritten"  # You can use 'trocr-base-stage1' or other variant

# -------------------------------------------------
# SETUP MLFLOW
# -------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# -------------------------------------------------
# DATA PREP (replace with your actual CSV or Dataset logic)
# -------------------------------------------------
# Example: Load your CSV into a Hugging Face Dataset
def load_receipt_dataset(csv_path="data/receipts.csv"):
    dataset = load_dataset("csv", data_files=csv_path, split="train")

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

    def preprocess(example):
        image = Image.open(example["image_path"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
        example["pixel_values"] = pixel_values
        example["labels"] = processor.tokenizer(example["text"], return_tensors="pt").input_ids[0]
        return example

    dataset = dataset.map(preprocess)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    return dataset, processor

# -------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------
def main():
    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        
        dataset, processor = load_receipt_dataset(csv_path="data/receipts.csv")
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

        training_args = Seq2SeqTrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=2,
            num_train_epochs=3,
            predict_with_generate=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=True if torch.cuda.is_available() else False,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")

        mlflow.log_metric("final_epoch", training_args.num_train_epochs)

        print("✅ Training complete and model logged to MLflow")

if __name__ == "__main__":
    main()
