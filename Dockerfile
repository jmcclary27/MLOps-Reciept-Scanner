FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r train_requirements.txt

# Run the training script by default
ENTRYPOINT ["python", "train_model.py"]
