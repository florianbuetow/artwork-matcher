FROM python:3.12-slim

WORKDIR /repo/services/storage

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY libs/service-commons/ /repo/libs/service-commons/
COPY services/storage/pyproject.toml services/storage/uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/storage/config.yaml .
COPY services/storage/src/ src/

EXPOSE 8004

CMD ["uv", "run", "uvicorn", "storage_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8004"]
