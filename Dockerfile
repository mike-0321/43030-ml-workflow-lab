# Configuration supplied for learning; not built or run in the recorded exercise.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY learning_service learning_service
COPY artifacts/digits_pipeline.joblib artifacts/digits_pipeline.joblib
COPY artifacts/manifest.json artifacts/manifest.json
# LAB_API_KEY must be supplied at runtime; no credential is embedded here.
CMD ["uvicorn", "learning_service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
