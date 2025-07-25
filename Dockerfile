FROM python:3.9-slim

WORKDIR /app

COPY . .
COPY vertex_dvc_key.json /app/vertex_dvc_key.json

RUN pip install --upgrade pip && \
    pip install -r train_requirements.txt && \
    pip install dvc && \
    pip install dvc-gs && \
    pip install gcsfs

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/vertex_dvc_key.json"

ENTRYPOINT ["bash", "-c", "dvc pull && python train_model.py --csv artifacts/data/receipt_dataset.csv"]

