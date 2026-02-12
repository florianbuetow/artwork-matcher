FROM python:3.12-slim

WORKDIR /repo/services/gateway

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY libs/service-commons/ /repo/libs/service-commons/
COPY services/gateway/pyproject.toml services/gateway/uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/gateway/config.yaml .
COPY services/gateway/src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
