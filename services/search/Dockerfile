FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY config.yaml .
COPY src/ src/

EXPOSE 8002

CMD ["uv", "run", "uvicorn", "search_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8002"]
