FROM python:3.11-slim

WORKDIR /app

# System deps needed for chromadb/tokenizers/spacy build steps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

COPY . .

# Ingest documents into Chroma at build time, so the deployed container
# starts with data already indexed - Render's free tier disk is ephemeral
# on redeploy, so baking the index into the image avoids an empty DB.
RUN python -c "from ingestion.loader import run_ingestion; print(run_ingestion())"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]