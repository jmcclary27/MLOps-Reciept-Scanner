import json
import mlflow
import shutil
from mlflow.artifacts import download_artifacts
from pathlib import Path

# === Load config ===
with open("mlflow_config.json", "r") as f:
    config = json.load(f)

MLFLOW_TRACKING_URI = config["MLFLOW_TRACKING_URI"]
run_id = config["RUN_ID"]

# === STEP 1: Set your remote MLflow tracking URI ===
# Replace this with the URI of your deployed MLflow server (e.g., Vertex-hosted)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
artifact_path = "model"
flat_dir = Path("saved_model")

# DOWNLOAD
if not flat_dir.exists():
    print(f"[*] Downloading model from run {run_id} ...")
    full_path = Path(download_artifacts(run_id=run_id, artifact_path=artifact_path))
else:
    full_path = flat_dir / "model"
    print("[!] 'saved_model/' already exists — using local copy.")

# FLATTEN
flat_dir.mkdir(exist_ok=True)
print("[*] Flattening structure...")

# Move config + model weights
shutil.copy(full_path / "model" / "config.json", flat_dir / "config.json")
shutil.copy(full_path / "model" / "generation_config.json", flat_dir / "generation_config.json")
for file in (full_path / "model").glob("*.safetensors*"):
    shutil.copy(file, flat_dir / file.name)

# Move tokenizer
for file in (full_path / "components/tokenizer").glob("*"):
    shutil.copy(file, flat_dir / file.name)

# Move image processor config
shutil.copy(full_path / "components/image_processor/preprocessor_config.json", flat_dir / "preprocessor_config.json")

print(f"[✓] Model ready in {flat_dir.resolve()} for from_pretrained()")
