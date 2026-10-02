# Base: runtime dependencies only, shared by the runtime and test images.
FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first for better layer caching.
COPY source/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Test image: adds dev dependencies and the test suite. `source/` is copied as
# a directory (not flattened) so `pythonpath = ["source"]` in pyproject.toml
# resolves identically here and in local development.
FROM base AS test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pyproject.toml .
COPY source/ source/
COPY tests/ tests/

# Runtime image: the gateway as a user runs it.
FROM base AS runtime

COPY source/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
