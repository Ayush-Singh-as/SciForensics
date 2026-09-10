# Backend image.
#
# The weights are NOT baked in: they are a 35 MB release asset fetched by
# `sciforensics.weights.ensure()` with SHA-256 verification, so the image stays
# small and a weights update does not require a rebuild. Mount a cache volume
# (see docker-compose.yml) to avoid re-downloading on every container start.
FROM python:3.12-slim AS base

# WeasyPrint needs Pango/cairo at runtime; opencv-python-headless needs libGL's
# ABI stubs even in headless mode. Installed in one layer, no recommends.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata first, so a source-only change does not reinstall torch.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/

# CPU-only torch: the CUDA wheels are ~2.5 GB and a demo container has no GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install ".[report,api]"

# Non-root. The API writes only to its own temp dir, which it creates itself.
RUN useradd --create-home --uid 10001 sci
USER sci

EXPOSE 8000

# The healthcheck hits /healthz, which deliberately does not touch the model,
# so a slow first inference cannot mark the container unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4).status==200 else 1)"

CMD ["uvicorn", "sciforensics.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
