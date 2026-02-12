FROM python:3.12-slim

WORKDIR /repo/services/geometric

# Install system dependencies for OpenCV headless
# libglib2.0-0: GLib (required by OpenCV)
# libgl1: OpenGL (sometimes needed)
# libxcb1, libx11-6: X11 libraries (needed even for headless in some cases)
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY libs/service-commons/ /repo/libs/service-commons/
COPY services/geometric/pyproject.toml services/geometric/uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/geometric/config.yaml .
COPY services/geometric/src/ src/

EXPOSE 8003

CMD ["uv", "run", "uvicorn", "geometric_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8003"]
