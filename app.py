from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io

app = Flask(__name__)

# Load model and processor
model = VisionEncoderDecoderModel.from_pretrained("saved_model")
processor = TrOCRProcessor.from_pretrained("saved_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Process image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"text": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
