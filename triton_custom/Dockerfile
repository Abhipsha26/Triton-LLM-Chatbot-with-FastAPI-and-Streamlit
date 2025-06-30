FROM nvcr.io/nvidia/tritonserver:24.03-py3

# Install Python packages and system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir torch transformers protobuf

# Optional: set environment variables for Python backend if needed
ENV PYTHONUNBUFFERED=1

