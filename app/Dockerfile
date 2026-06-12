FROM python:3.11-slim-bookworm

# uv binary from the official image - pnned version for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.8.21 /uv /uvx /bin/

# PySpark requires Java
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Use the image's system Python, don't let uv download another
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install dependencies first — this layer caches as long as the lockfile is unchanged
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Then copy the app and finalize
COPY . .
RUN uv sync --locked --no-dev

# Put the venv on PATH so `streamlit` resolves
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]