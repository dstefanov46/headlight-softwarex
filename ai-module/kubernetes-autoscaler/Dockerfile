# Use a lightweight official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    gcc \
    && KUBECTL_VERSION="$(curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location https://dl.k8s.io/release/stable.txt)" \
    && curl --proto '=https' --tlsv1.2 --fail --show-error --location --remote-name "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/ \
    && pip install --no-cache-dir --only-binary :all: -r requirements.txt \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && rm -rf /var/lib/apt/lists/*

# Copy only the runtime files needed by the autoscaler image.
COPY entrypoint.sh .
COPY hpa.yaml .

RUN chmod +x entrypoint.sh
USER appuser

# Default command to run the script
CMD ["./entrypoint.sh"]
