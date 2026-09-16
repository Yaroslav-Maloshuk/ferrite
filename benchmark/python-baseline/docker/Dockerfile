FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.lock.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock.txt

FROM python:3.12-slim
WORKDIR /srv
COPY --from=builder /install /usr/local
COPY server.py .
EXPOSE 8081
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8081"]
