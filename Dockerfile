FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY saved_model ./saved_model

EXPOSE 8080
CMD ["python", "app.py"]
