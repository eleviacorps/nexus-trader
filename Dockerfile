FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-prod.txt /tmp/requirements-prod.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements-prod.txt

COPY config ./config
COPY src ./src
RUN mkdir -p /app/models/tft /app/outputs/evaluation

EXPOSE 8000

CMD ["python", "-m", "src.service.app"]
