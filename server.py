# server.py
import os, io, json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# ---- Load config from JSON file in the repo ----
CONFIG_PATH = os.getenv("VERTEX_CONFIG_PATH", "vertex_config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _cfg = json.load(f)

REGION      = _cfg["region"]
PROJECT_ID  = _cfg["project_id"]
ENDPOINT_ID = _cfg["endpoint_id"]

# Service account JSON for server-to-Vertex auth (env var must point to a key file)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Minimal upload guardrails
MAX_BYTES   = int(os.getenv("MAX_UPLOAD_BYTES", 6_000_000))  # ~6MB
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)

def get_token() -> str:
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_APPLICATION_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    return creds.token

@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    f = request.files["file"]
    filename = secure_filename(f.filename or "upload")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"Unsupported file type {ext}"), 415

    raw = f.read()
    if not raw or len(raw) > MAX_BYTES:
        return jsonify(error="Empty file or too large"), 413

    token = get_token()
    url = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:rawPredict"
    )

    files = {"file": (filename, io.BytesIO(raw), "application/octet-stream")}
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, files=files, timeout=120)

    if r.status_code != 200:
        return jsonify(error="Vertex call failed", status=r.status_code, details=r.text), 502

    return r.json(), 200

if __name__ == "__main__":
    # Dev run (for production, run with gunicorn/uvicorn/etc.)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8081")))
