FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# state/ holds the SQLite file built from the CSVs; data/ is mounted at runtime (see docker-compose.yml)
RUN useradd --create-home appuser && mkdir -p /srv/state /srv/data && chown -R appuser /srv/state
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('MCP_PORT', '8000'), timeout=3)"

CMD ["python", "-m", "app.server"]
