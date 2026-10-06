# DEVIATIONS — 논문 / 공개 코드 / 이식 결정 로그

근거 우선순위(PLAN §0.2): **논문 수치·동작 → (없거나 비현실적이면) 공개 코드 → (둘 다 없으면) 사용자 확인**.
태그: `[P]` 논문, `[C]` 공개 코드, `[U]` 사용자 결정, `[N]` 이식 때문에 새로 설계.
"위치"는 이 저장소에서 해당 결정이 구현된 곳이다.

| ID | 항목 | 논문 | 공개 코드 | 이식 결정 | 근거 | 위치 |
|---|---|---|---|---|---|---|
| D-01 | Variant 생성 | 자동(TensorRT/Neuron) | 수동 `profile_model.sh` + `.config` | 자동. `register_model` → 해당 GPU 타입 worker에서 `ProfileVariant` 실행. static 모드에서는 GPU 6장이 모두 worker에 잡혀 있어 별도 Job을 쓸 수 없으므로, 논문의 "target hardware를 가진 VM에서 프로파일" 문구대로 worker에서 수행 | P, N | `controller/registrar.py`, `worker/profiler.py` |
| D-02 | Variant 차원 | 아키텍처 × 프레임워크 × optimizer × batch × HW | 같음 | **(모델, GPU 타입)**, FP32 PyTorch, batch 1 → 129개 | U C3·C4 | `common/naming.py` |
| D-03 | 추론 지연 | profiled latency | batch {1,4,8} 선형회귀, perf client 평균 | batch 1 고정이라 회귀 불필요. **전처리 포함 worker 처리 시간**의 동시성-1 평균(30회) | U C6, C | `worker/profiler.py` |
| D-04 | 로드 지연 | "loading latency" | 복사 → `MODEL_READY` | 로드 시작부터 서빙 가능 시점까지(warmup 10회 포함). 요청이 실제로 기다리는 시간이 이것이기 때문 | C, N | `worker/runtime.py`, `profiler.py` |
| D-05 | 포화 처리량 Q_ij | 사용(Table 2, §4.2.1), 측정법 없음 | `1000/lat × batch` | 폐루프 4 클라이언트 × 5초 실측 | N | `worker/profiler.py` |
| D-06 | Active variant 여러 개 | 명시 없음 | `SLO − lat` 최소 | 코드 규칙. 동률이면 저가 GPU | C | `policy/selection.py` |
| D-07 | 요청 → worker (L3) | least-loaded | min-QPS worker | **bin packing**(Best-Fit, 인스턴스 부하 < Q/1.05 중 가장 찬 것). 여유가 전혀 없으면 least-loaded | U G4 (§4.2.3, §6.3) | `selection.pack_request` |
| D-08 | 새 인스턴스 배치 (L6) | lowest-util worker | min CPU∩GPU util + shuffle | **bin packing**(Best-Fit, GPU 메모리 1 GB slack, SM util < 80%). Inactive = 어느 worker에도 없는 variant | U G4, P | `selection.pack_placement` |
| D-09 | 로드 중인 인스턴스 | 없음 | 같은 worker로 가면 합류 | 로딩 중인 인스턴스가 있으면 그쪽으로 보내 합류(중복 로드 방지) | N | `selection.get_variant` |
| D-10 | 쓸 수 있는 variant 없음 | 명시 없음 | blacklist된 variant라도 보냄 | 가장 덜 찬 running 인스턴스로 fallback | C | `selection.get_variant` |
| D-11 | Interfered 완화 | 같은 worker의 다른 자원 → 없으면 least-loaded worker로 | **미구현**(RPC 정의만) | GPU 1장/Pod라 같은 worker의 다른 자원은 없음 → controller가 least-loaded worker에 로드 후 원래 인스턴스 언로드 | P | `monitor._mitigate`, `placement.py` |
| D-12 | Interfered 판정 | "profiled보다 높은 지연" | 계수 1.5 / min(5,15/lat), QPS > 0.3×용량, 해제 1.25, 큐 조건 | 계수·해제·QPS 하한은 코드. 큐 조건은 논문의 Overloaded(QPS ≥ peak)와 겹쳐 제외 | P, C | `policy/states.py` |
| D-13 | 지연 통계 | "inference latencies" | 로드 시간 포함 | 로드 시간 제외(콜드 로드는 추론 지연이 아님) | P | `worker/executor.py` |
| D-14 | 모니터링 주기 | 2 s | qps 1 s, 자원 2.5 s | 2 s. 창 안에서 로드된 인스턴스는 로드 이후 시간으로 QPS 계산 | P, N | `worker/monitor.py` |
| D-15 | GPU utilization | 정의 없음 | GPU **메모리** 사용률 | **NVML SM 사용률**(0.25 s 샘플의 창 평균). 메모리는 배치 제약(3)에만 사용 | U G3 | `worker/monitor.py` |
| D-16 | CPU utilization | – | `/proc/stat`(호스트 전체) | Pod cgroup(노드당 Pod 2개) | N | `worker/monitor.py` |
| D-17 | 비용 C_ij | AWS 가격, §6.2 "메모리 footprint 비례" | worker 내부 결정에 가격 없음 | `HW_COST[hw] × peak GB`, 비율 1:2:4, λ = 1.0 | P §6.2, U G1·G2 | `policy/costs.py` |
| D-18 | Model-Autoscaler | ILP → greedy, slack 1.05 | `loadHeuristic 0.0002`, `sum_cost > 1.5×` 규칙 | headroom < 1.05면 replicate / upgrade(더 빠른 GPU)를 ILP 목적함수로 비교. 코드 규칙은 쓰지 않음 | P | `policy/scaling.py` |
| D-19 | 복제 위치 | worker 내부 | `GPU_MAX_REPLICAS = 1` | worker당 1개 → 복제·upgrade·downgrade는 모두 다른 worker에. worker가 결정하고 controller가 배치 | C, P | `placement.proto`, `controller/placement.py` |
| D-20 | Scale-down 대기 | T_v = 로드 지연 | 연속 20회(GPU) | T_v(로드 지연, 1 s 슬롯 단위). 로드 직후 첫 창이 끝나기 전에는 판단하지 않음 | P, C(보호 조건) | `scaling.ScaleDownTimer`, `worker/autoscaler.py` |
| D-21 | 마지막 인스턴스 제거 | 명시 없음 | QPS 0이면 scale down | 코드 규칙(T_v 후) | C | `scaling.scale_down_option` |
| D-22 | Executor blacklist | **없음**(과부하는 Overloaded 상태로 처리) | util > 80 → 2 s, 초당 200 req → 2 s | 구현했지만 **기본 off**. 논문은 같은 기능을 상태 머신으로 처리하고, G3로 SM 사용률을 쓰면 바쁜 GPU가 2초마다 blacklist되어 요청 경로 로드를 유발. `EXEC_BLACKLIST_*_ENABLED`로 켤 수 있음 — **사용자 확인 필요** | P 우선 | `common/config.py`, `dispatcher.py`, `vm_autoscaler.py` |
| D-23 | 결정 모드 6 | – | shuffle, blacklist 생략, VM flag | 동률 shuffle + Interfered 후보 시 VM flag는 채택. "blacklist 검사 생략"은 논문과 충돌 → 검사 유지 | C, P (G7) | `selection.get_variant`, `dispatcher.py` |
| D-24 | VM 규칙 | 규칙 1–3, 80% | util/flag 기반, HW 3종 고정 매핑 | 규칙을 GPU 타입별로 평가. 규칙 3의 타입 = Overloaded가 가장 많은 타입(최대치면 더 빠른 타입). scale-down 조건(8%/5%, 평균 8%/30%, 15회)은 코드 값을 타입별로 적용 | P, C, G5 | `scaling.vm_scale_up/down_ok` |
| D-25 | VM backoff | – | 15 × 2 s | 코드 값 | C | `controller/vm_autoscaler.py` |
| D-26 | Worker 운용 | 필요할 때 VM 추가 | 같음 | **static(기본, Lumina처럼 GPU마다 worker 상시)** / **dynamic(논문)** 둘 다. `WORKER_MODE`로 전환 | U | `k8s/configmap.yaml`, `scripts/deploy.sh` |
| D-27 | 동적 worker 생성 | EC2 | `start_vm.sh` | worker Deployment의 pod template으로 bare Pod 생성(`infaas-mode: dynamic`, ownerReference). static과 같은 pod spec | N | `controller/k8s_adapter.py` |
| D-28 | 장애 복구 | heartbeat + 상태 복원 | heartbeat만 | Pod watch + heartbeat. 실패한 worker의 variant를 같은 타입의 다음 worker에 재적재 | P §7 | `vm_autoscaler.sync` |
| D-29 | Metadata Store 위치 | controller와 같은 머신 | 같은 VM의 redis | controller Pod의 sidecar(같은 노드, localhost) | P | `k8s/controller.yaml` |
| D-30 | Model Repository | 고용량 영구 저장소 | S3 → 로컬 캐시 | 노드 hostPath HF 캐시(Lumina와 같은 가중치), `HF_HUB_OFFLINE=1` | N | `docker/worker.Dockerfile` |
| D-31 | worker 프로세스 구성 | executor와 monitor는 별도 프로세스 | 같은 프로세스의 스레드 | 코드처럼 스레드(카운터 공유) | C | `worker/main.py` |
| D-32 | 공개 API | Table 4 | queryfe.proto | 원 API + Lumina 호환 `podexec.Executor`(:8081). `dispatch=predict`는 `PREDICT_PROXY`로 Lumina 서버에 전달 | N | `controller/frontend.py` |
| D-33 | 범위 밖 | accuracy/appID, offline, NLP | 구현됨 | 필드만 유지 | U C1 | `proto/*.proto` |
| D-34 | 중복 스케일 방지 | – | `model_available_` 플래그 | `<v>-pending`(요청자가 NX로 설정, controller가 완료 시 해제, TTL 30 s) | C, N | `redis_metadata.set_pending` |
| D-35 | 공통 워밍업 | – | – | 트레이스 앞 30 s는 재생하되 채점 제외(두 시스템 동일) | U C8, G6 | `bench/run_trace.py` |
| D-36 | 배치 결정 동시성 | – | – | Placement는 한 번에 하나(일관된 스냅샷) | N | `controller/placement.py` |
| D-37 | 인스턴스 실행 스레드 | – | Triton model instance(자체 실행 스레드) | 인스턴스마다 전용 실행 스레드에서 forward. 여러 gRPC 스레드가 번갈아 forward하면 cuDNN이 스레드별 핸들·plan cache를 다시 만들어 ConvNeXt forward가 7 ms → 2.2 s(2080 Ti 실측). 로드 시 warmup도 이 스레드에서 돌아 비용을 로드에 포함 | C(Triton 의미), N | `worker/runtime.py` |
| D-38 | 프로파일 이미지 | – | – | Lumina 루트의 FHD `frame_1080p.jpg`(1920×1080) — 실험 이미지와 동일해야 함 | U | `infaas.cli.register --image` |
