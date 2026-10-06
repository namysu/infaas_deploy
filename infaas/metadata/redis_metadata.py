"""Metadata Store (paper §3.2, §5) on Redis.

Port of INFaaS src/metadata-store/redis_metadata.{h,cc}: same key names and
suffixes wherever the concept still exists, so a key seen here can be looked up
in the original. Dropped: CPU/Inferentia executor flags, slack/exclusive GPU,
accuracy bins, parent scaledown (PLAN §1.6, §0.4). Added: an explicit
{variant, worker} state hash (paper §5 "each model-variant instance's state is
encoded as a {variant, worker} pair"), per-worker GPU memory, the executor's GPU
type, and a pending-placement marker.

Keys (C = as in the original, N = new):
  allexecutors                SET  executors                                   C ALLEXEC_SET
  <w>-info                    HASH addr hw mem_free mem_total util cpu_util hb managed   N
  <w>-blist                   STR  executor blacklist, EXPIRE                  C BLIST_SUFF
  <w>-modvar                  SET  variants on the executor                    C EXECMVAR_SUFF
  gpuutil_set / cpuutil_set   ZSET executor -> utilization                     C
  vmscale-<hw>                STR  VM-scale request for a GPU type             C VMSCALE_KEY (+hw)
  recover-<hw>                LIST variants of failed workers to restore       N (paper §7)
  model_set                   SET  parent models                               C MODEL_SET
  modelvar_set                SET  model-variants                              C MODELVAR_SET
  <parent>-modvariants        SET  variants of a parent                        C MODVAR_SUFF
  <v>-info                    HASH profile                                     C MODINFO_SUFF
  <v>-parent                  STR  parent                                      C PARENT_SUFF
  <parent>b1inflat / loadlat / b1totlat  ZSET variant -> ms                    C INFLAT/LOADLAT/TOTLAT
  registry_version            INT  bumped on every registration                N
  <v>-state                   HASH worker -> state                             N
  <v>-modqps                  ZSET worker -> qps                               C MODQPS_SUFF
  <v>-modavglat               ZSET worker -> avg latency                       C MODAVGLAT_SUFF
  allrunning                  SET  variants with an instance                   C RUNMODS_SET
  <v>-pending                 STR  placement in progress, EXPIRE               N
"""
from __future__ import annotations

import json
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import redis

from infaas.common import config
from infaas.common.naming import parse_variant
from infaas.policy import states
from infaas.policy.types import InstanceView, Snapshot, VariantProfile, WorkerView

ALLEXEC_SET = "allexecutors"
MODEL_SET = "model_set"
MODELVAR_SET = "modelvar_set"
RUNMODS_SET = "allrunning"
CPUUTIL_SET = "cpuutil_set"
GPUUTIL_SET = "gpuutil_set"
REGISTRY_VERSION = "registry_version"


def _k(*parts: str) -> str:
    return "-".join(parts)


def connect(host: str = None, port: int = None) -> redis.Redis:
    return redis.Redis(host=host or config.REDIS_HOST, port=port or config.REDIS_PORT,
                       decode_responses=True, socket_keepalive=True)


def profile_from_hash(v: str, h: Dict[str, str]) -> VariantProfile:
    model, hw = parse_variant(v)
    return VariantProfile(variant=v, model=h.get("model", model), hw=h.get("hw", hw),
                          lat_ms=float(h["inf_latency_ms"]),
                          load_ms=float(h["load_latency_ms"]),
                          sat_qps=float(h["sat_qps"]),
                          mem_bytes=int(float(h["peak_memory_bytes"])))


class RedisMetadata:
    def __init__(self, client: Optional[redis.Redis] = None) -> None:
        self.r = client if client is not None else connect()

    # ------------------------------------------------------------ executors
    def add_executor(self, name: str, addr: str, hw: str, managed: bool = False) -> None:
        p = self.r.pipeline()
        p.sadd(ALLEXEC_SET, name)
        p.hset(_k(name, "info"), mapping={"addr": addr, "hw": hw,
                                         "managed": int(managed), "hb": time.time()})
        p.zadd(GPUUTIL_SET, {name: 0.0}, nx=True)
        p.zadd(CPUUTIL_SET, {name: 0.0}, nx=True)
        p.execute()

    def delete_executor(self, name: str) -> List[str]:
        """Remove an executor and every instance on it. Returns those variants."""
        variants = sorted(self.r.smembers(_k(name, "modvar")))
        for v in variants:
            self.remove_running_model(name, v)
        p = self.r.pipeline()
        p.srem(ALLEXEC_SET, name)
        p.delete(_k(name, "info"), _k(name, "blist"), _k(name, "modvar"))
        p.zrem(GPUUTIL_SET, name)
        p.zrem(CPUUTIL_SET, name)
        p.execute()
        return variants

    def get_all_executors(self) -> List[str]:
        return sorted(self.r.smembers(ALLEXEC_SET))

    def executor_exists(self, name: str) -> bool:
        return bool(self.r.sismember(ALLEXEC_SET, name))

    def get_executor_info(self, name: str) -> Dict[str, str]:
        return self.r.hgetall(_k(name, "info"))

    def get_executor_addr(self, name: str) -> str:
        return self.r.hget(_k(name, "info"), "addr") or ""

    def blacklist_executor(self, name: str, expire_s: int = config.EXEC_BLACKLIST_EXPIRE_S) -> None:
        self.r.set(_k(name, "blist"), 1, ex=max(1, int(expire_s)))

    def is_blacklisted(self, name: str) -> bool:
        return bool(self.r.exists(_k(name, "blist")))

    def update_worker_stats(self, name: str, util: float, cpu_util: float,
                            mem_free: Optional[int], mem_total: Optional[int]) -> None:
        """Memory left out when it could not be measured (keeps the last value)."""
        fields = {"util": util, "cpu_util": cpu_util, "hb": time.time()}
        if mem_free is not None and mem_total:
            fields.update(mem_free=mem_free, mem_total=mem_total)
        p = self.r.pipeline()
        p.hset(_k(name, "info"), mapping=fields)
        p.zadd(GPUUTIL_SET, {name: util})
        p.zadd(CPUUTIL_SET, {name: cpu_util})
        p.execute()

    # VM-scale requests, one flag per GPU type
    def set_vm_scale(self, hw: str) -> None:
        self.r.set(_k("vmscale", hw), 1)

    def unset_vm_scale(self, hw: Optional[str] = None) -> None:
        hws = [hw] if hw else config.HW_TYPES
        self.r.delete(*[_k("vmscale", h) for h in hws])

    def vm_scale_flags(self) -> List[str]:
        hws = list(config.HW_TYPES)
        vals = self.r.mget([_k("vmscale", h) for h in hws])
        return [h for h, v in zip(hws, vals) if v]

    def push_recover(self, hw: str, variants: Iterable[str]) -> None:
        vs = list(variants)
        if vs:
            self.r.rpush(_k("recover", hw), *vs)

    def pop_recover(self, hw: str) -> List[str]:
        p = self.r.pipeline()
        p.lrange(_k("recover", hw), 0, -1)
        p.delete(_k("recover", hw))
        vs, _ = p.execute()
        return list(dict.fromkeys(vs))

    # ------------------------------------------------------------ registry
    def add_parent_model(self, parent: str) -> None:
        self.r.sadd(MODEL_SET, parent)

    def parent_model_registered(self, parent: str) -> bool:
        return bool(self.r.sismember(MODEL_SET, parent))

    def model_registered(self, variant: str) -> bool:
        return bool(self.r.sismember(MODELVAR_SET, variant))

    def get_all_parent_models(self) -> List[str]:
        return sorted(self.r.smembers(MODEL_SET))

    def add_model(self, profile: dict) -> str:
        """Register one profiled variant (the original add_model)."""
        v, parent = profile["variant"], profile["model"]
        def enc(val):
            if isinstance(val, (dict, list)):
                return json.dumps(val)
            if isinstance(val, bool):            # Redis takes no bools
                return int(val)
            return val
        info = {k: enc(val) for k, val in profile.items() if val is not None}
        info["hw"] = profile["hardware"]
        lat, load = float(profile["inf_latency_ms"]), float(profile["load_latency_ms"])
        p = self.r.pipeline()
        p.sadd(MODEL_SET, parent)
        p.sadd(MODELVAR_SET, v)
        p.sadd(_k(parent, "modvariants"), v)
        p.set(_k(v, "parent"), parent)
        p.delete(_k(v, "info"))
        p.hset(_k(v, "info"), mapping=info)
        p.zadd(parent + "b1inflat", {v: lat})
        p.zadd(parent + "loadlat", {v: load})
        p.zadd(parent + "b1totlat", {v: load + lat})
        p.incr(REGISTRY_VERSION)
        p.execute()
        return v

    def delete_model(self, variant: str) -> None:
        parent = self.r.get(_k(variant, "parent"))
        p = self.r.pipeline()
        p.srem(MODELVAR_SET, variant)
        if parent:
            p.srem(_k(parent, "modvariants"), variant)
            for suf in ("b1inflat", "loadlat", "b1totlat"):
                p.zrem(parent + suf, variant)
        p.delete(_k(variant, "info"), _k(variant, "parent"))
        p.incr(REGISTRY_VERSION)
        p.execute()

    def get_all_model_variants(self, parent: str) -> List[str]:
        return sorted(self.r.smembers(_k(parent, "modvariants")))

    def inf_lat_bin(self, parent: str, min_lat: float, max_lat: float,
                    max_results: int = 5) -> List[str]:
        return self.r.zrangebyscore(parent + "b1inflat", min_lat, max_lat,
                                    start=0, num=max_results)

    def get_model_info(self, variant: str) -> Dict[str, str]:
        return self.r.hgetall(_k(variant, "info"))

    def get_profile(self, variant: str) -> Optional[VariantProfile]:
        h = self.get_model_info(variant)
        return profile_from_hash(variant, h) if h else None

    def registry_version(self) -> int:
        return int(self.r.get(REGISTRY_VERSION) or 0)

    def load_registry(self) -> Tuple[int, Dict[str, List[VariantProfile]]]:
        """All parents with their variant profiles (static data, cached by callers)."""
        version = self.registry_version()
        parents = self.get_all_parent_models()
        p = self.r.pipeline()
        for parent in parents:
            p.smembers(_k(parent, "modvariants"))
        members = p.execute()
        p = self.r.pipeline()
        flat = [(parent, v) for parent, vs in zip(parents, members) for v in sorted(vs)]
        for _, v in flat:
            p.hgetall(_k(v, "info"))
        out: Dict[str, List[VariantProfile]] = {parent: [] for parent in parents}
        for (parent, v), h in zip(flat, p.execute()):
            if h:
                out[parent].append(profile_from_hash(v, h))
        return version, out

    # ------------------------------------------------------------ instances
    def set_instance_state(self, worker: str, variant: str, state: str) -> None:
        p = self.r.pipeline()
        p.hset(_k(variant, "state"), worker, state)
        p.sadd(_k(worker, "modvar"), variant)
        p.sadd(RUNMODS_SET, variant)
        p.execute()

    def remove_running_model(self, worker: str, variant: str) -> None:
        p = self.r.pipeline()
        p.hdel(_k(variant, "state"), worker)
        p.srem(_k(worker, "modvar"), variant)
        p.zrem(_k(variant, "modqps"), worker)
        p.zrem(_k(variant, "modavglat"), worker)
        p.hlen(_k(variant, "state"))
        left = p.execute()[-1]
        if not left:
            self.r.srem(RUNMODS_SET, variant)

    def get_instance_states(self, variant: str) -> Dict[str, str]:
        return self.r.hgetall(_k(variant, "state"))

    def get_variants_on_executor(self, worker: str) -> List[str]:
        return sorted(self.r.smembers(_k(worker, "modvar")))

    def is_model_running(self, variant: str, worker: str = "") -> bool:
        st = self.get_instance_states(variant)
        if worker:
            return st.get(worker) in states.RUNNING
        return any(s in states.RUNNING for s in st.values())

    def update_instance(self, worker: str, variant: str, state: str,
                        qps: float, avg_lat: float) -> None:
        """One monitoring-window update of an instance (qpsMonitor)."""
        p = self.r.pipeline()
        p.hset(_k(variant, "state"), worker, state)
        p.sadd(_k(worker, "modvar"), variant)
        p.sadd(RUNMODS_SET, variant)
        p.zadd(_k(variant, "modqps"), {worker: qps})
        p.zadd(_k(variant, "modavglat"), {worker: avg_lat})
        p.execute()

    def get_model_qps(self, worker: str, variant: str) -> float:
        return float(self.r.zscore(_k(variant, "modqps"), worker) or 0.0)

    def min_qps_name(self, variant: str, max_results: int = 3) -> List[str]:
        return self.r.zrangebyscore(_k(variant, "modqps"), "-inf", "+inf",
                                    start=0, num=max_results)

    def cluster_stats(self, variants: Sequence[str]) -> Dict[str, Tuple[int, float]]:
        """variant -> (running instances, total qps) across all workers."""
        p = self.r.pipeline()
        for v in variants:
            p.hgetall(_k(v, "state"))
            p.zrange(_k(v, "modqps"), 0, -1, withscores=True)
        res = p.execute()
        out = {}
        for i, v in enumerate(variants):
            st, qps = res[2 * i], dict(res[2 * i + 1])
            running = [w for w, s in st.items() if s in states.RUNNING]
            out[v] = (len(running), sum(qps.get(w, 0.0) for w in running))
        return out

    def set_pending(self, variant: str, ttl_s: float = config.PENDING_TTL_S) -> bool:
        """Mark a placement in progress. False if one already is."""
        return bool(self.r.set(_k(variant, "pending"), 1, nx=True,
                               ex=max(1, int(ttl_s))))

    def clear_pending(self, variant: str) -> None:
        self.r.delete(_k(variant, "pending"))

    def is_pending(self, variant: str) -> bool:
        return bool(self.r.exists(_k(variant, "pending")))

    # ------------------------------------------------------------ snapshots
    def snapshot(self, variants: Sequence[str], executors: Sequence[str]) -> Snapshot:
        """Everything one dispatch decision needs, in a single round trip."""
        p = self.r.pipeline()
        for w in executors:
            p.hmget(_k(w, "info"), "hw", "addr", "util", "mem_free", "mem_total")
            p.exists(_k(w, "blist"))
        for v in variants:
            p.hgetall(_k(v, "state"))
            p.zrange(_k(v, "modqps"), 0, -1, withscores=True)
        res = p.execute()
        snap = Snapshot()
        for i, w in enumerate(executors):
            hw, addr, util, free, total = res[2 * i]
            if not hw:
                continue
            snap.workers[w] = WorkerView(name=w, hw=hw, addr=addr or "",
                                         util=float(util or 0.0),
                                         mem_free=int(float(free or 0)),
                                         mem_total=int(float(total or 0)),
                                         blacklisted=bool(res[2 * i + 1]))
        base = 2 * len(executors)
        for j, v in enumerate(variants):
            st, qps = res[base + 2 * j], dict(res[base + 2 * j + 1])
            snap.instances[v] = [InstanceView(variant=v, worker=w, state=s,
                                              qps=float(qps.get(w, 0.0)))
                                 for w, s in sorted(st.items())]
        return snap

    def worker_snapshot(self) -> Snapshot:
        """All executors and every instance (placement / VM-Autoscaler view)."""
        executors = self.get_all_executors()
        variants = sorted(self.r.smembers(RUNMODS_SET))
        return self.snapshot(variants, executors)

    # ------------------------------------------------------------ housekeeping
    def flush_dynamic(self) -> None:
        """Forget executors and instances, keep the registry (controller start)."""
        for w in self.get_all_executors():
            self.delete_executor(w)
        keys = []
        for pattern in ("*-state", "*-modqps", "*-modavglat", "*-pending",
                        "*-blist", "vmscale-*", "recover-*", "*-modvar", "*-info"):
            for k in self.r.scan_iter(match=pattern, count=1000):
                keys.append(k)
        # keep variant profiles (<v>-info); executors' <w>-info were deleted above
        variants = set(self.r.smembers(MODELVAR_SET))
        keys = [k for k in keys if not (k.endswith("-info") and k[:-5] in variants)]
        if keys:
            self.r.delete(*keys)
        self.r.delete(RUNMODS_SET, GPUUTIL_SET, CPUUTIL_SET)
