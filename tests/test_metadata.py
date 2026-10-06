"""Metadata Store schema on fakeredis."""
import fakeredis
import pytest

from infaas.metadata.redis_metadata import RedisMetadata
from infaas.policy import states


def _profile(v, lat=20.0, load=1000.0):
    model, hw = v.split("__")
    return {"variant": v, "model": model, "hardware": hw, "inf_latency_ms": lat,
            "load_latency_ms": load, "sat_qps": 50.0, "peak_memory_bytes": 500_000_000,
            "stage_ms": {"forward": 5.0}, "dali": True, "image_size": [1920, 1080]}


@pytest.fixture
def md():
    return RedisMetadata(fakeredis.FakeRedis(decode_responses=True))


def test_registry_roundtrip(md):
    md.add_model(_profile("resnet-50__a30", 20))
    md.add_model(_profile("resnet-50__2080ti", 40))
    assert md.parent_model_registered("resnet-50")
    assert md.get_all_model_variants("resnet-50") == ["resnet-50__2080ti", "resnet-50__a30"]
    assert md.inf_lat_bin("resnet-50", 0, 30) == ["resnet-50__a30"]
    v, reg = md.load_registry()
    assert v == 2 and {p.variant for p in reg["resnet-50"]} == {"resnet-50__a30", "resnet-50__2080ti"}
    p = md.get_profile("resnet-50__a30")
    assert p.hw == "a30" and p.lat_ms == 20.0 and p.mem_bytes == 500_000_000


def test_instances_and_snapshot(md):
    md.add_model(_profile("resnet-50__a30"))
    md.add_executor("x0", "10.0.0.1:9000", "a30")
    md.update_worker_stats("x0", 42.0, 10.0, 20 << 30, 24 << 30)
    md.update_instance("x0", "resnet-50__a30", states.ACTIVE, 12.5, 21.0)
    s = md.snapshot(["resnet-50__a30"], md.get_all_executors())
    assert s.workers["x0"].util == 42.0 and s.workers["x0"].mem_free == 20 << 30
    (inst,) = s.of("resnet-50__a30")
    assert (inst.worker, inst.state, inst.qps) == ("x0", states.ACTIVE, 12.5)
    assert md.cluster_stats(["resnet-50__a30"]) == {"resnet-50__a30": (1, 12.5)}
    assert md.min_qps_name("resnet-50__a30") == ["x0"]


def test_delete_executor_cleans_instances(md):
    md.add_executor("x0", "a:1", "a30")
    md.update_instance("x0", "resnet-50__a30", states.ACTIVE, 1, 1)
    assert md.delete_executor("x0") == ["resnet-50__a30"]
    assert md.get_instance_states("resnet-50__a30") == {}
    assert md.get_all_executors() == []


def test_flags_pending_blacklist(md):
    md.set_vm_scale("a30")
    assert md.vm_scale_flags() == ["a30"]
    md.unset_vm_scale()
    assert md.vm_scale_flags() == []
    assert md.set_pending("v") and not md.set_pending("v")
    md.clear_pending("v")
    assert md.set_pending("v")
    md.add_executor("x0", "a:1", "a30")
    md.blacklist_executor("x0")
    assert md.is_blacklisted("x0")


def test_recover_queue(md):
    md.push_recover("a30", ["v1", "v2", "v1"])
    assert md.pop_recover("a30") == ["v1", "v2"]
    assert md.pop_recover("a30") == []


def test_flush_dynamic_keeps_registry(md):
    md.add_model(_profile("resnet-50__a30"))
    md.add_executor("x0", "a:1", "a30")
    md.update_instance("x0", "resnet-50__a30", states.ACTIVE, 1, 1)
    md.set_vm_scale("a30")
    md.flush_dynamic()
    assert md.get_all_executors() == []
    assert md.get_instance_states("resnet-50__a30") == {}
    assert md.vm_scale_flags() == []
    assert md.get_profile("resnet-50__a30") is not None
