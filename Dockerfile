FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-prod.txt /tmp/requirements-prod.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements-prod.txt

COPY config ./config
COPY scripts ./scripts
COPY src ./src
RUN mkdir -p /app/models/tft /app/outputs/evaluation /app/outputs/live /app/outputs/v25 /app/outputs/deployment /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
