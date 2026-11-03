# Voice agent server image (Sprint 5). Built and run by docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Kolkata \
    HOST=0.0.0.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes rebuild in seconds instead of minutes
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY app ./app
COPY scripts ./scripts
COPY config ./config
COPY tests ./tests

# Run as a normal user, not root
RUN useradd --no-user-group --create-home --uid 1000 voice && mkdir -p /app/logs && chown -R voice /app/logs
USER voice

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read() == b'ok' else 1)"

CMD ["python", "-m", "app.server"]
