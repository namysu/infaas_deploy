### 1. 서버 GPU 확인

```
deploy/docker/infaas-docker.sh gpus

이후 deploy/docker/infaas-docker.env 수정

GPUS=0:a30,1:a30,2:a5000,3:a5000      # nvidia-smi 번호:타입 (gpus 출력의 suggested 줄을 그대로 써도 됨)
HW_COST=a5000:2,a30:4                 # GPUS에 쓴 타입마다 상대 가격 (타입이 하나면 생략 가능)
MODEL_DIR=/data/models        # 가중치 캐시 경로
PROFILE_DIR=/data/infaas-profiles
LOG_DIR=/data/infaas-logs
```

포트(50052, 50053, 8081)가 사용중인지 확인

### 2. 이미지 빌드 및 weight 다운로드

(weight의 경우 model download 시간을 피하기 위함이며, 현재는 43개 모델 - 예측모델 만들때 썼던 목록임)
```
deploy/docker/infaas-docker.sh build        # infaas-controller:0.1.0, infaas-worker:0.1.0
deploy/docker/infaas-docker.sh download     # 43개 모델 → MODEL_DIR (약 6 GB, 받은 건 건너뜀)

```

### 3. deploy
```
deploy/docker/infaas-docker.sh up
deploy/docker/infaas-docker.sh state # state에서 worker가 올라왔는지 확인
```

### 4. 요청 보내기

타입이 2개 (gRPC, API) 구현되어있고, INFaaS의 기본 통신 방식은 API 임 (우리 기법과의 비교를 위해 gRPC를 구현해둔거)

__4.1 gRPC 예시__
```python
import grpc
from infaas.proto import podexec_pb2, podexec_pb2_grpc

stub = podexec_pb2_grpc.ExecutorStub(grpc.insecure_channel("<호스트>:8081"))
img = open("goldfish.jpg", "rb").read()               # JPEG 그대로 (base64 아님)
r = stub.Infer(podexec_pb2.InferRequest(model="resnet-50", image=img, slo=200.0), timeout=120)
print("rejected" if r.rejected else "ok", r.gpu, r.inference, f"{r.latency:.1f} ms")

```

__4.2 gRPC__
```
python3 -m infaas.cli.online_query --controller <호스트>:50052 --model resnet-50 --image goldfish.jpg --slo 200

```

### 5. logging & undeploy
```
deploy/docker/infaas-docker.sh state --watch 2           # variant 상태 (ACTIVE / OVERLOADED / INTERFERED)
deploy/docker/infaas-docker.sh logs infaas-worker-0
deploy/docker/infaas-docker.sh logs infaas-vm-autoscaler
deploy/docker/infaas-docker.sh down                      # 중지 (프로파일·로그는 호스트 경로에 남음)
```