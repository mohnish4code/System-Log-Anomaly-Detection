# One container: FastAPI serves both the API (/api, /docs) and the built React UI (/).
# Trained artifacts, data and results are mounted, not baked in:
#   docker compose up --build
# or without compose:
#   docker build -t log-anomaly .
#   docker run -p 8000:8000 -v ./artifacts:/app/artifacts -v ./data:/app/data -v ./results:/app/results log-anomaly

FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Runtime dependencies only; test and lint tools are in requirements-dev.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app src ./src
COPY --from=frontend --chown=app:app /frontend/dist ./frontend/dist
RUN mkdir -p artifacts data results && chown app:app artifacts data results
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
