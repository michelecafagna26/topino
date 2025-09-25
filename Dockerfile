FROM python:3.13.5-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser \
    && mkdir -p /home/appuser/.cache/uv \
    && chown -R appuser:appuser /home/appuser/.cache \
    && chown -R appuser:appuser /app

USER appuser

COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./
COPY src/ ./src/

RUN uv sync

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
ENV STREAMLIT_DISABLE_METRICS=1

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/app.py"] 