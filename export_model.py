import os
import json
import mlflow

from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# === Load config ===
with open("mlflow_config.json", "r") as f:
    config = json.load(f)

MLFLOW_TRACKING_URI = config["MLFLOW_TRACKING_URI"]
RUN_ID = config["RUN_ID"]
FINAL_DIR = "saved_model"

# === Set MLflow URI ===
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# === Full download path for the model
download_path = os.path.join(os.getcwd(), FINAL_DIR)

# === Download model directly into ./saved_model/
print(f"📦 Downloading model to: {download_path}")
mlflow.artifacts.download_artifacts(f"runs:/{RUN_ID}/model", dst_path=download_path)

print(f"✅ Model successfully saved in: {FINAL_DIR}/")
