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

def _normalize_root_uri(gs_or_local_uri: str) -> str:
    """Ensure we list/copy from the parent 'artifacts' folder even if given .../model or .../components."""
    u = gs_or_local_uri.rstrip("/")
    for suffix in ("/model", "/components"):
        if u.endswith(suffix):
            return u[: -len(suffix)]
    return u

def _download_and_flatten_to(local_dir: str, src_uri: str):
    """
    Copy *all files* found under:
      artifacts/model/** and artifacts/components/** (including image_processor/, tokenizer/)
    into 'local_dir' FLAT (1 level), i.e., /model/<filename>.
    If filenames collide, later files overwrite earlier ones (expected to be unique for HF artifacts).
    """
    os.makedirs(local_dir, exist_ok=True)
    # If something already exists, assume we've done this once (idempotent startup)
    if any(True for _ in os.scandir(local_dir)):
        return

    root_uri = _normalize_root_uri(src_uri)

    if root_uri.startswith("gs://"):
        bucket_name, prefix = _parse_gs_uri(root_uri)
        client = storage.Client()
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            # skip "directory" placeholders
            if blob.name.endswith("/"):
                continue

            rel = blob.name[len(prefix):].lstrip("/")   # e.g., "model/config.json" or "components/tokenizer/vocab.json"
            if not rel:
                continue

            parts = rel.split("/")

            # Only keep files that are under model/ or components/
            if parts[0] not in ("model", "components"):
                continue

            # Flatten completely to the basename
            dst = os.path.join(local_dir, os.path.basename(rel))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            blob.download_to_filename(dst)
    else:
        # Local folder source; copy recursively and flatten
        for dirpath, _dirnames, filenames in os.walk(root_uri):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), root_uri)
                parts = rel.replace("\\", "/").split("/")
                if parts[0] not in ("model", "components"):
                    continue
                dst = os.path.join(local_dir, os.path.basename(fn))
                src = os.path.join(dirpath, fn)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                # copy binary-safe
                with open(src, "rb") as s, open(dst, "wb") as d:
                    d.write(s.read())

def _ensure_loaded():
    global MODEL, PROCESSOR, READY

    if READY and MODEL is not None and PROCESSOR is not None:
        return

    # Always pull from the parent 'artifacts' so we get both model/ and components/
    src = _normalize_root_uri(AIP_STORAGE_URI)

    _download_and_flatten_to(LOCAL_MODEL_DIR, src)

    model_path = LOCAL_MODEL_DIR  # now contains config.json, pytorch_model.bin, preprocessor_config.json, tokenizer files, etc.

    # Basic sanity checks (helpful error if something’s missing)
    need = ["config.json", "pytorch_model.bin", "preprocessor_config.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        raise FileNotFoundError(f"Missing required files in {model_path}: {missing}")

    # Load
    PROCESSOR = TrOCRProcessor.from_pretrained(model_path)
    MODEL = VisionEncoderDecoderModel.from_pretrained(model_path).to(DEVICE)
    MODEL.eval()
    READY = True

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200


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
    port = int(os.environ.get("PORT", 8080))
    # Fine for testing; consider gunicorn in production
    app.run(host="0.0.0.0", port=8080)
