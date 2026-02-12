FROM python:3.12-slim

WORKDIR /repo/services/search

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY libs/service-commons/ /repo/libs/service-commons/
COPY services/search/pyproject.toml services/search/uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/search/config.yaml .
COPY services/search/src/ src/

EXPOSE 8002

CMD ["uv", "run", "uvicorn", "search_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8002"]
