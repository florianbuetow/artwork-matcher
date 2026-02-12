FROM python:3.12-slim

WORKDIR /repo/services/embeddings

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY libs/service-commons/ /repo/libs/service-commons/
COPY services/embeddings/pyproject.toml services/embeddings/uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/embeddings/config.yaml .
COPY services/embeddings/src/ src/

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "embeddings_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
