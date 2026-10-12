FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    PORT=8000 \
    HARBOR_BUNDLE=/app/models/california-housing

WORKDIR /app

# Pinned dependencies first, so this layer is cached between builds.
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE NOTICE /app/
COPY src /app/src
RUN pip install --no-deps .

# Train the demo bundle at build time. Same data, seed and versions give the same model.
COPY data /app/data
RUN harbor train data/california_housing.csv \
      --target median_house_value \
      --descriptions data/california_housing.columns.json \
      --out models/california-housing

COPY entrypoint.sh /app/
RUN useradd --create-home --uid 10001 harbor
USER harbor

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
