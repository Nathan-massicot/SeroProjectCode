# 🧠 SeroProjectCode – Data Analysis Pipeline

Data analysis for a health app.

## Local Python setup

Prerequisites:
- Python `3.12.x`
- `uv`

Example:
```bash
python3 --version
uv sync
uv run streamlit run Streamlit/Loading_Page.py
```

Local app URL:
- `http://localhost:8501`

## Docker

### 1) Prepare environment file

```bash
cp .env.example .env
```

Edit `.env` if needed:
- `HOST_PORT` public host port (default `8501`)
- `PORT` container Streamlit port (default `8501`)
- `SERO_ALLOW_CLEAN_EXPORT` default `0`

### 2) Build production image (multi-stage target)

```bash
docker build --target prod -t sero-streamlit:prod .
```

### 3) Run with Docker Compose (recommended)

```bash
docker compose up -d --build
```

Open:
- `http://localhost:${HOST_PORT}`

### 4) Health verification

```bash
# Streamlit internal health endpoint
curl -fsS "http://localhost:${HOST_PORT}/_stcore/health"

# Compose service health status
docker compose ps
```

### 5) Logs

```bash
docker compose logs -f app
```

## What is configured

- Multi-stage Dockerfile with `dev` and `prod` targets.
- `prod` runs as non-root user `app`.
- Strict dependency install with `uv sync --frozen`.
- Dynamic port support with `${PORT:-8501}`.
- Healthcheck enabled on `/_stcore/health`.
- Persistent Hugging Face model cache volume only.
- Runtime-only files in image (`Streamlit/`, `src/`).
- Sensitive data excluded from build context (`data/` in `.dockerignore`).

## Deployment checklist (for server handoff)

1. Confirm server architecture: `amd64` or `arm64`.
2. Confirm runtime model: Docker Compose, Kubernetes, or managed platform.
3. Confirm public exposure: direct port or reverse proxy + HTTPS.
4. Set `.env` values (`HOST_PORT`, `PORT`, optional vars).
5. Run `docker compose up -d --build`.
6. Validate health endpoint and service status.
7. Confirm model cache volume exists and persists between restarts.
8. Confirm logs are collected from stdout/stderr.

## Troubleshooting

1. App not reachable:
- Check `docker compose ps` and mapped `HOST_PORT`.
- Check logs: `docker compose logs -f app`.

2. Slow first startup:
- First run downloads models.
- Next runs reuse persisted `sero_hf_cache` volume.

3. Port conflict:
- Change `HOST_PORT` in `.env`, then restart Compose.

4. Rebuild after dependency changes:
- Update `pyproject.toml`/`uv.lock`, then run `docker compose up -d --build`.
