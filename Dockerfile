# Pin Python to a patch release for reproducible builds.
FROM python:3.12.12-slim AS base

# Set Python and uv behavior defaults used by all stages.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    APP_HOME=/app
ENV STREAMLIT_APP=Streamlit/Loading_Page.py

# Set the working directory for subsequent instructions.
WORKDIR /app

# Install minimal OS packages needed for HTTPS downloads and health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv so dependency resolution/install is fast and lockfile-aware.
RUN pip install --no-cache-dir uv

# Copy only dependency metadata first to maximize layer cache reuse.
COPY pyproject.toml uv.lock ./

# Create the virtual environment from the lockfile without installing the project package.
RUN uv sync --frozen --no-dev --no-install-project

# ------------------------------
# Development target
# ------------------------------
# Start a development-oriented image from the prepared base.
FROM base AS dev

# Keep the virtualenv binaries first in PATH for direct command usage.
ENV PATH="/app/.venv/bin:${PATH}"

# Copy source code used by the application during local development.
COPY Streamlit ./Streamlit
COPY src ./src

# Expose the Streamlit default container port for local runs.
EXPOSE 8501

# Start Streamlit in headless server mode with dynamic PORT support.
CMD ["sh", "-c", "streamlit run ${STREAMLIT_APP} --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]

# ------------------------------
# Production target
# ------------------------------
# Start a production-oriented image from the same prepared base.
FROM base AS prod

# Create an unprivileged runtime user for better container security.
RUN useradd --create-home --home-dir /home/app --shell /usr/sbin/nologin app

# Ensure the app runtime and model cache directories exist and are writable by non-root user.
RUN mkdir -p /home/app/.cache/sero_hf /app \
    && chown -R app:app /home/app /app

# Set runtime environment values for cache paths and optional app behavior.
ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/home/app/.cache/sero_hf \
    HF_HUB_CACHE=/home/app/.cache/sero_hf \
    HUGGINGFACE_HUB_CACHE=/home/app/.cache/sero_hf \
    TRANSFORMERS_CACHE=/home/app/.cache/sero_hf \
    SERO_ALLOW_CLEAN_EXPORT=0 \
    PORT=8501 \
    STREAMLIT_APP=Streamlit/Loading_Page.py

# Copy only runtime source files and set non-root ownership directly.
COPY --chown=app:app Streamlit ./Streamlit
COPY --chown=app:app src ./src

# Switch to the non-root runtime user.
USER app

# Declare the default Streamlit container port.
EXPOSE 8501

# Define a container healthcheck against Streamlit's internal health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

# Start Streamlit in headless server mode with explicit address and dynamic port.
CMD ["sh", "-c", "streamlit run ${STREAMLIT_APP} --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
