# Dockerfile

FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-0

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install extra packages if needed
RUN pip install --upgrade pip
RUN pip install transformers datasets pillow mlflow google-cloud-storage gcsfs

ENTRYPOINT ["python", "src/models/train_model.py"]
