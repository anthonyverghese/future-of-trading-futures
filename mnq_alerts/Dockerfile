# Dockerfile — MNQ Alert System
# ================================
# Used for running the app in AWS ECS/Fargate or on an EC2 instance via Docker.
#
# Build:  docker build -t mnq-alerts .
# Run:    docker run --env-file .env mnq-alerts

FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies first (separate layer for better Docker cache reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY *.py .

# Ensure Python output is streamed to CloudWatch Logs / docker logs in real time
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
