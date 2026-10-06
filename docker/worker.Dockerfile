# INFaaS GPU worker. Same base and ML stack as Lumina's worker
# (new_podexecutor/docker/worker.Dockerfile @8eb588c) so both systems run the
# same torch/transformers/DALI [PLAN A-03]; plus redis (Metadata Store client)
# and nvidia-ml-py (SM utilization, [U G3]).
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir --break-system-packages \
        transformers \
        accelerate \
        safetensors \
        pillow \
        "grpcio>=1.81" \
        grpcio-health-checking \
        "protobuf>=6.33" \
        redis \
        nvidia-ml-py

RUN pip install --no-cache-dir --break-system-packages \
        --extra-index-url https://pypi.nvidia.com \
        nvidia-dali-cuda120

# Weights come from the node's HF cache (/models, the hostPath Lumina uses).
# HF_HUB_OFFLINE keeps a load from asking the Hub for updates: the Model
# Repository is local, and a load sits on the request path in INFaaS.
ENV HF_HOME=/models \
    TORCH_HOME=/models \
    HF_HUB_OFFLINE=1 \
    GPU_TYPE=unknown \
    WORKER_PORT=9000 \
    PYTHONUNBUFFERED=1 \
    CUDA_MODULE_LOADING=EAGER

WORKDIR /app
COPY infaas /app/infaas

EXPOSE 9000
CMD ["python", "-m", "infaas.worker.main"]
