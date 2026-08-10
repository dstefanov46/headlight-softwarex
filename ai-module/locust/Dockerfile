# Use a lightweight Python image
FROM locustio/locust:2.29.1

ARG KUBECTL_VERSION=v1.32.0

# Switch to root for package installation
USER root

# Install system dependencies and kubectl.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    && curl --proto '=https' --tlsv1.2 --fail --show-error --location --remote-name "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy load test files
COPY load_generator.py /app/load_generator.py
COPY user_generator.req /app/user_generator.req
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x entrypoint.sh
USER locust

# Default command to run the worload generator
CMD ["./entrypoint.sh"]

