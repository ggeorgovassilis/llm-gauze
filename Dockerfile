FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY source/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY source/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
