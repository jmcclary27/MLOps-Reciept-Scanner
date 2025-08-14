import os
import io
from urllib.parse import urlparse

from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Only extra dependency for GCS download:
from google.cloud import storage

app = Flask(__name__)

# Vertex sets this when you upload with --artifact-uri=gs://...
AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "saved_model")
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/model")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = None
PROCESSOR = None
READY = False


def _parse_gs_uri(gs_uri: str):
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"AIP_STORAGE_URI must be gs://... ; got: {gs_uri}")
    parsed = urlparse(gs_uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _download_artifacts_if_needed(gs_uri: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    # If something already exists, assume we previously downloaded
    if any(True for _ in os.scandir(local_dir)):
        return

    bucket_name, prefix = _parse_gs_uri(gs_uri)
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    found = False
    for blob in blobs:
        rel_path = blob.name[len(prefix):].lstrip("/")
        if not rel_path:
            continue
        found = True
        dst = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        blob.download_to_filename(dst)

    if not found:
        raise FileNotFoundError(f"No model files found at {gs_uri}")


def _ensure_loaded():
    global MODEL, PROCESSOR, READY

    if READY and MODEL is not None and PROCESSOR is not None:
        return

    # If artifacts are in GCS, download them once
    if AIP_STORAGE_URI.startswith("gs://"):
        _download_artifacts_if_needed(AIP_STORAGE_URI, LOCAL_MODEL_DIR)
        model_path = LOCAL_MODEL_DIR
    else:
        model_path = AIP_STORAGE_URI  # e.g., "saved_model" baked into the image

    PROCESSOR = TrOCRProcessor.from_pretrained(model_path)
    MODEL = VisionEncoderDecoderModel.from_pretrained(model_path).to(DEVICE)
    MODEL.eval()

    READY = True


@app.route("/health", methods=["GET"])
def health():
    # Only return 200 after the model is fully ready
    try:
        _ensure_loaded()
        return "ok", 200
    except Exception as e:
        return f"loading: {e}", 503


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        _ensure_loaded()
        image = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
        with torch.no_grad():
            pixel_values = PROCESSOR(images=image, return_tensors="pt").pixel_values.to(DEVICE)
            generated_ids = MODEL.generate(
                pixel_values,
                max_length=128,
                num_beams=2,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
            text = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return jsonify({"text": text}), 200
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Fine for testing; consider gunicorn in production
    app.run(host="0.0.0.0", port=8080)
