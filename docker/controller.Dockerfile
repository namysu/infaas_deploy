# INFaaS controller: Front-End/Dispatcher/Registrar/Placement, the VM-Autoscaler,
# and the Redis Metadata Store (run from this image as a sidecar, so it sits on
# the controller's machine as in paper §3.2 without pulling another image).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends redis-server \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
        "grpcio>=1.81" \
        grpcio-health-checking \
        "protobuf>=6.33" \
        redis \
        kubernetes

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY infaas /app/infaas

EXPOSE 50052 50053 50054 8081 16379
CMD ["python", "-m", "infaas.controller.main"]
