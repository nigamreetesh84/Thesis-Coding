# One image for every node (SuperLink + bank SuperNodes); the role is chosen by the command.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# CPU build of PyTorch first (much smaller than the CUDA wheels), then the project requirements
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt "flwr>=1.45,<2"
COPY fl fl
COPY src src
COPY evaluation evaluation
COPY config config
