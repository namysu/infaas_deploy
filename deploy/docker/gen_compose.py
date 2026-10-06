#!/usr/bin/env python3
"""Write a docker compose deployment of INFaaS from deploy/docker/infaas-docker.env.

One worker container per GPU listed in GPUS (nvidia-smi index or UUID), bound to
that GPU alone, plus Redis (Metadata Store), the controller and the VM-Autoscaler.
The VM-Autoscaler finds the workers through ORCHESTRATOR=static and the
STATIC_WORKERS list written here (infaas/controller/static_workers.py).

Outputs, in --out (default deploy/docker/generated/):
  docker-compose.yml   the services (JSON, which is valid YAML: no PyYAML needed)
  infaas.env           INFaaS settings shared by all services (env_file)

    python3 deploy/docker/gen_compose.py                 # generate
    python3 deploy/docker/gen_compose.py --suggest       # list GPUs, suggest a GPUS line
Standard library only, so it runs on any host with python3.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent

# keys consumed here; everything else in the env file is passed to INFaaS as-is
DEPLOY_KEYS = {
    "GPUS", "HW_TYPES", "HW_COST", "PREPROCESS_MODES", "CPUSETS", "MODEL_DIR", "PROFILE_DIR",
    "LOG_DIR", "CONTROLLER_IMAGE", "WORKER_IMAGE", "NETWORK", "PUBLISH_ADDR",
    "WORKER_BASE_PORT", "ROLE", "CONTROLLER_HOST", "ADVERTISE_HOST", "EXTRA_WORKERS",
    "WORKER_PREFIX", "HOST_PORT_OFFSET",
}
# keys this script sets per service or derives; the env file must not set them
RESERVED = {
    "ORCHESTRATOR", "STATIC_WORKERS", "MAX_WORKERS", "REDIS_HOST", "REDIS_PORT",
    "CONTROLLER_ADDR", "WORKER_NAME", "GPU_TYPE", "PREPROCESS_MODE", "WORKER_PORT",
    "POD_NAME", "DYN_INIT_WORKERS",
}
REDIS_PORT = 16379
PORTS = {"queryfe": 50052, "modelreg": 50053, "placement": 50054, "compat": 8081}
HEALTH = ("import grpc,sys;from grpc_health.v1 import health_pb2 as h,health_pb2_grpc as g;"
          "sys.exit(0 if g.HealthStub(grpc.insecure_channel('127.0.0.1:{port}'))"
          ".Check(h.HealthCheckRequest(),timeout=3).status==1 else 1)")


class ConfigError(Exception):
    pass


@dataclass
class Gpu:
    device: str            # what docker gets in device_ids: index or UUID
    index: Optional[int]   # nvidia-smi index, when known
    hw: str
    name: str = ""         # nvidia-smi product name, when known


# ------------------------------------------------------------ reading
def read_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, eq, val = line.partition("=")
        if not eq:
            raise ConfigError(f"{path}:{n}: expected KEY=VALUE")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        out[key.strip()] = val
    return out


def parse_pairs(spec: str, what: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        k, sep, v = item.rpartition(":")
        if not sep or not k or not v:
            raise ConfigError(f"{what}: {item!r} is not key:value")
        out.append((k.strip(), v.strip()))
    return out


def parse_map(spec: str, what: str) -> Dict[str, str]:
    pairs = parse_pairs(spec, what)
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ConfigError(f"{what}: a key appears twice ({spec})")
    return dict(pairs)


def nvidia_smi() -> Optional[List[dict]]:
    """GPUs as nvidia-smi lists them, or None when it is not available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in res.stdout.strip().splitlines():
        idx, uuid, name, mem = (c.strip() for c in line.split(",", 3))
        gpus.append({"index": int(idx), "uuid": uuid, "name": name, "mem_mib": int(float(mem))})
    return gpus


def gpu_type_from_name(name: str) -> str:
    """A short type name from an nvidia-smi product name ("NVIDIA A30" -> "a30")."""
    n = name.lower()
    for word in ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx"):
        n = n.replace(word, " ")
    m = re.search(r"(\d{4})\s*ti\b", n)
    if m:
        return f"{m.group(1)}ti"
    m = re.search(r"\b([ahlvtb]\d{1,4}s?)\b", n)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{4})\b", n)
    if m:
        return m.group(1)
    return re.sub(r"[^a-z0-9]", "", n) or "gpu"


def parse_gpus(spec: str, smi: Optional[List[dict]], verify: bool) -> Tuple[List[Gpu], List[str]]:
    warnings: List[str] = []
    spec = spec.strip()
    if not spec:
        return [], warnings
    if spec.lower() == "auto":
        if smi is None:
            raise ConfigError("GPUS=auto needs nvidia-smi on this host; list the GPUs explicitly")
        return [Gpu(str(g["index"]), g["index"], gpu_type_from_name(g["name"]), g["name"])
                for g in smi], warnings
    by_index = {g["index"]: g for g in smi or []}
    by_uuid = {g["uuid"]: g for g in smi or []}
    gpus: List[Gpu] = []
    for key, hw in parse_pairs(spec, "GPUS"):
        if key.isdigit():
            idx = int(key)
            g = by_index.get(idx)
            if smi is not None and g is None:
                raise ConfigError(f"GPUS: no GPU with nvidia-smi index {idx} "
                                  f"(have {sorted(by_index)})")
        elif key.startswith(("GPU-", "MIG-")):
            g = by_uuid.get(key)
            if smi is not None and g is None and key.startswith("GPU-"):
                raise ConfigError(f"GPUS: no GPU with UUID {key}")
            idx = g["index"] if g else None
        else:
            raise ConfigError(f"GPUS: {key!r} is neither an nvidia-smi index nor a UUID")
        name = g["name"] if g else ""
        if g and gpu_type_from_name(name) != hw:
            warnings.append(f"GPU {key} is '{name}' but GPUS calls it {hw!r} (check the type)")
        gpus.append(Gpu(key, idx, hw, name))
    keys = [g.index if g.index is not None else g.device for g in gpus]
    if len(set(keys)) != len(keys):
        raise ConfigError("GPUS lists the same GPU twice")
    if smi is None and verify:
        warnings.append("nvidia-smi not found: GPU indices were not checked")
    return gpus, warnings


def parse_workers(spec: str) -> List[Tuple[str, str, str]]:
    """EXTRA_WORKERS "name=hw@host:port" -> [(name, hw, host:port)]."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, eq, rest = item.partition("=")
        hw, at, addr = rest.partition("@")
        host, colon, port = addr.rpartition(":")
        if not (eq and at and colon and name and hw and host and port.isdigit()):
            raise ConfigError(f"EXTRA_WORKERS: {item!r} is not name=gpu_type@host:port")
        out.append((name.strip(), hw.strip(), addr.strip()))
    return out


# ------------------------------------------------------------ building
def esc(v) -> str:
    """Compose interpolates $VAR; keep values literal."""
    return str(v).replace("$", "$$")


def build(cfg: Dict[str, str], smi: Optional[List[dict]], verify: bool = True):
    clash = sorted(RESERVED & set(cfg))
    if clash:
        raise ConfigError(f"set by gen_compose.py, remove from the env file: {clash}")
    if cfg.get("WORKER_MODE", "static").lower() != "static":
        raise ConfigError("docker runs a fixed worker list: WORKER_MODE must be static")

    role = cfg.get("ROLE", "all").lower()
    network = cfg.get("NETWORK", "bridge").lower()
    if role not in ("all", "controller", "workers"):
        raise ConfigError(f"ROLE={role!r}: all | controller | workers")
    if network not in ("bridge", "host"):
        raise ConfigError(f"NETWORK={network!r}: bridge | host")
    if role == "workers" and network != "host":
        raise ConfigError("ROLE=workers needs NETWORK=host (other hosts must reach the workers)")
    if role == "workers" and not cfg.get("CONTROLLER_HOST"):
        raise ConfigError("ROLE=workers needs CONTROLLER_HOST")

    gpus, warnings = parse_gpus(cfg.get("GPUS", ""), smi, verify) if role != "controller" else ([], [])
    extra = parse_workers(cfg.get("EXTRA_WORKERS", "")) if role != "workers" else []
    if role != "controller" and not gpus:
        raise ConfigError("GPUS is empty")
    if role != "workers" and not gpus and not extra:
        raise ConfigError("no workers: set GPUS or EXTRA_WORKERS")

    cost = parse_map(cfg.get("HW_COST", ""), "HW_COST")
    used = {g.hw for g in gpus} | {hw for _, hw, _ in extra}
    missing = sorted(used - set(cost))
    if missing:
        if len(used) == 1 and not cost:
            cost = {next(iter(used)): "1"}
        else:
            raise ConfigError(f"HW_COST has no price for {missing}")
    if cfg.get("HW_TYPES", "").strip():
        hw_types = [h.strip() for h in cfg["HW_TYPES"].split(",") if h.strip()]
        if sorted(used - set(hw_types)):
            raise ConfigError(f"HW_TYPES lacks {sorted(used - set(hw_types))}")
    else:
        hw_types = sorted(used, key=lambda h: (float(cost[h]), h))
    if len(used) == 1:
        warnings.append(f"one GPU type ({next(iter(used))}): variants cannot move between "
                        "GPU types, so INFaaS has no model-vertical scaling here")

    modes = parse_map(cfg.get("PREPROCESS_MODES", ""), "PREPROCESS_MODES")
    cpusets = parse_map(cfg.get("CPUSETS", ""), "CPUSETS")
    base_port = int(cfg.get("WORKER_BASE_PORT", "9000"))
    prefix = cfg.get("WORKER_PREFIX") or (
        f"infaas-worker-{socket.gethostname().split('.')[0]}" if role == "workers" else "infaas-worker")

    bridge = network == "bridge"
    remote = role == "controller" or bool(extra)          # workers elsewhere need redis + placement
    if bridge:
        redis_host, ctrl_host = "infaas-redis", "infaas-controller"
    elif role == "workers":
        redis_host = ctrl_host = cfg["CONTROLLER_HOST"]
    else:
        redis_host = ctrl_host = "127.0.0.1"
    net = {"networks": ["infaas"]} if bridge else {"network_mode": "host"}
    common = {"restart": "unless-stopped", "env_file": ["infaas.env"], **net}
    redis_dep = {"depends_on": {"infaas-redis": {"condition": "service_healthy"}}} \
        if role != "workers" else {}
    services: Dict[str, dict] = {}

    # ---- workers on this host
    local: List[Tuple[str, str, str]] = []
    for pos, g in enumerate(gpus):
        num = g.index if g.index is not None else pos
        name = f"{prefix}-{num}"
        port = base_port if bridge else base_port + num
        svc = {
            "image": esc(cfg.get("WORKER_IMAGE", "infaas-worker:0.1.0")),
            "container_name": name,
            "hostname": name,
            **common,
            "environment": {
                "WORKER_NAME": name, "GPU_TYPE": g.hw, "WORKER_PORT": str(port),
                "PREPROCESS_BACKEND": "auto",
                "PREPROCESS_MODE": modes.get(g.hw, "cpu_decode_gpu"),
                "REDIS_HOST": redis_host, "REDIS_PORT": str(REDIS_PORT),
                "CONTROLLER_ADDR": f"{ctrl_host}:{PORTS['placement']}",
            },
            # worker load/unload/state events go to the same log dir as the controller's
            "volumes": [f"{esc(cfg.get('MODEL_DIR', '/data/podexec-models'))}:/models",
                        f"{esc(cfg.get('LOG_DIR', '/data/infaas-logs'))}:/logs"],
            "shm_size": "1g",
            "deploy": {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "device_ids": [g.device], "capabilities": ["gpu"]}]}}},
            "healthcheck": {"test": ["CMD", "python", "-c", HEALTH.format(port=port)],
                            "interval": "5s", "timeout": "5s", "retries": 3,
                            "start_period": "180s"},
            **redis_dep,
        }
        cs = cpusets.get(str(g.index)) if g.index is not None else cpusets.get(g.device)
        if cs:
            svc["cpuset"] = cs
        services[name] = svc
        addr_host = name if bridge else ("127.0.0.1" if role == "all" else cfg.get("ADVERTISE_HOST", ""))
        local.append((name, g.hw, f"{addr_host}:{port}"))

    workers = local + extra
    names = [w[0] for w in workers]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate worker names: {names}")

    # ---- controller side
    if role != "workers":
        ctrl_img = esc(cfg.get("CONTROLLER_IMAGE", "infaas-controller:0.1.0"))
        bind_redis = "0.0.0.0" if (bridge or remote) else "127.0.0.1"
        redis = {
            "image": ctrl_img, "container_name": "infaas-redis", **net,
            "restart": "unless-stopped",
            "command": ["redis-server", "--port", str(REDIS_PORT), "--bind", bind_redis,
                        "--protected-mode", "no", "--save", "", "--appendonly", "no"],
            "healthcheck": {"test": ["CMD", "redis-cli", "-p", str(REDIS_PORT), "ping"],
                            "interval": "2s", "timeout": "2s", "retries": 30},
        }
        ctrl_env = {"REDIS_HOST": redis_host, "REDIS_PORT": str(REDIS_PORT),
                    "PROFILE_DIR": "/profiles", "PREDICT_PROXY": ""}
        controller = {
            "image": ctrl_img, "container_name": "infaas-controller", **common,
            "command": ["python", "-m", "infaas.controller.main"],
            "environment": dict(ctrl_env),
            "volumes": [f"{esc(cfg.get('PROFILE_DIR', '/data/infaas-profiles'))}:/profiles",
                        f"{esc(cfg.get('LOG_DIR', '/data/infaas-logs'))}:/logs"],
            "healthcheck": {"test": ["CMD", "python", "-c", HEALTH.format(port=PORTS["queryfe"])],
                            "interval": "5s", "timeout": "5s", "retries": 6, "start_period": "10s"},
            **redis_dep,
        }
        autoscaler = {
            "image": ctrl_img, "container_name": "infaas-vm-autoscaler", **common,
            "command": ["python", "-m", "infaas.controller.vm_autoscaler"],
            "environment": {"REDIS_HOST": redis_host, "REDIS_PORT": str(REDIS_PORT)},
            "volumes": [f"{esc(cfg.get('LOG_DIR', '/data/infaas-logs'))}:/logs"],
            **redis_dep,
        }
        if bridge:
            pub = cfg.get("PUBLISH_ADDR", "0.0.0.0")
            off = int(cfg.get("HOST_PORT_OFFSET", "0") or 0)
            controller["ports"] = [f"{pub}:{PORTS[k] + off}:{PORTS[k]}"
                                   for k in ("queryfe", "modelreg", "compat")]
            redis["ports"] = [f"127.0.0.1:{REDIS_PORT + off}:{REDIS_PORT}"]
            if remote:
                controller["ports"].append(
                    f"{pub}:{PORTS['placement'] + off}:{PORTS['placement']}")
                redis["ports"] = [f"{pub}:{REDIS_PORT + off}:{REDIS_PORT}"]
        services = {"infaas-redis": redis, "infaas-controller": controller,
                    "infaas-vm-autoscaler": autoscaler, **services}

    compose = {"name": "infaas", "services": services}
    if bridge:
        compose["networks"] = {"infaas": {"name": "infaas"}}

    max_workers: Dict[str, int] = {}
    for _, hw, _ in workers:
        max_workers[hw] = max_workers.get(hw, 0) + 1
    env = {k: v for k, v in cfg.items() if k not in DEPLOY_KEYS}
    env.update({
        "HW_TYPES": ",".join(hw_types),
        "HW_COST": ",".join(f"{h}:{cost[h]}" for h in hw_types),
        "MAX_WORKERS": ",".join(f"{h}:{n}" for h, n in sorted(max_workers.items())),
        "ORCHESTRATOR": "static",
        "WORKER_MODE": "static",
        "STATIC_WORKERS": ",".join(f"{n}={hw}@{a}" for n, hw, a in workers),
    })
    info = {"role": role, "network": network, "gpus": gpus, "workers": workers,
            "local": local, "hw_types": hw_types,
            "ports": services.get("infaas-controller", {}).get("ports", [])}
    return compose, env, warnings, info


def write(out: Path, compose: dict, env: Dict[str, str], src: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    header = (f"# generated by deploy/docker/gen_compose.py from {src} — do not edit;\n"
              "# change the env file and regenerate. JSON is valid YAML.\n")
    (out / "docker-compose.yml").write_text(header + json.dumps(compose, indent=2) + "\n")
    lines = [f"# generated by deploy/docker/gen_compose.py from {src}"]
    lines += [f"{k}={v}" for k, v in env.items()]
    (out / "infaas.env").write_text("\n".join(lines) + "\n")


def suggest(smi: Optional[List[dict]]) -> int:
    if smi is None:
        print("nvidia-smi not found on this host")
        return 1
    print(f"{'index':>5}  {'type':8s} {'memory':>8}  name / uuid")
    for g in smi:
        print(f"{g['index']:>5}  {gpu_type_from_name(g['name']):8s} {g['mem_mib']:>6}MiB  "
              f"{g['name']}  {g['uuid']}")
    line = ",".join(f"{g['index']}:{gpu_type_from_name(g['name'])}" for g in smi)
    print(f"\nsuggested:\nGPUS={line}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=str(HERE / "infaas-docker.env"))
    ap.add_argument("--out", default=str(HERE / "generated"))
    ap.add_argument("--suggest", action="store_true", help="list GPUs and suggest a GPUS line")
    ap.add_argument("--no-verify", action="store_true",
                    help="do not check GPUS against nvidia-smi (generate on another host)")
    args = ap.parse_args()

    smi = None if args.no_verify else nvidia_smi()
    if args.suggest:
        return suggest(nvidia_smi())
    src = Path(args.env)
    try:
        compose, env, warnings, info = build(read_env(src), smi, verify=not args.no_verify)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    write(Path(args.out), compose, env, src)

    for w in warnings:
        print(f"warning: {w}")
    print(f"role={info['role']} network={info['network']} gpu types={info['hw_types']}")
    for g in info["gpus"]:
        print(f"  GPU {g.device:>3} -> {g.hw:8s} {g.name}")
    for n, hw, a in info["workers"]:
        print(f"  worker {n:28s} {hw:8s} {a}")
    if info["ports"]:
        print(f"  controller ports (host:container): {', '.join(info['ports'])}")
    if info["role"] == "workers":
        line = ",".join(f"{n}={hw}@{a}" for n, hw, a in info["local"])
        print(f"\nadd to EXTRA_WORKERS on the controller host:\n{line}")
    model_dir = Path(read_env(src).get("MODEL_DIR", "/data/podexec-models"))
    if info["role"] != "controller" and not (model_dir.is_dir() and any(model_dir.iterdir())):
        print(f"warning: MODEL_DIR {model_dir} is missing or empty on this host "
              "(infaas-docker.sh download)")
    print(f"wrote {Path(args.out) / 'docker-compose.yml'} and infaas.env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
