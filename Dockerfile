# Multi-stage build
# 1. Build Frontend
FROM node:18-alpine as build-step
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# 2. Build Backend & Serve
FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt ./
# Install system dependencies (OpenCV)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY backend/ ./backend
COPY --from=build-step /app/frontend/dist ./frontend/dist

ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production

EXPOSE 8080

# Run gunicorn
# We need to make sure app.py is set up to serve static files from /frontend/dist
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "backend.app:app"]
