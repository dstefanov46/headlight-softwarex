# Match the supported runtime used by the graph-free LSTM image and tests.
FROM python:3.11.9-slim-bookworm

ARG KUBECTL_VERSION=v1.32.0

# Set working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker caching. Build with
# ai-module/ as the Docker context.
COPY ai-module/requirements.txt .

# Install system dependencies, kubectl, and Python libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    gcc \
    && curl --proto '=https' --tlsv1.2 --fail --show-error --location --remote-name "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --only-binary :all: --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 \
    && pip install --no-cache-dir --only-binary :all: -r requirements.txt

# Copy local code and experiment config used by the runtime scripts.
COPY ai-module/ .
COPY metrics/fetch_service_metrics.py .
COPY metrics/SLA_thresholds.json .
COPY configs/application_services.json .
COPY configs/additional_services.json .
COPY configs/clusters.json .
COPY configs/service_profiles/ /app/service_profiles/

RUN chmod +x entrypoint.sh \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default command to run the script
CMD ["./entrypoint.sh"]
