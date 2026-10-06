"""Algorithm 1 (paper §4.1) with bin packing [U G4]."""
from conftest import A, GB, I, L, O, snap, worker

from infaas.policy import selection


def test_no_variant_meets_slo_rejects_with_suggestion(resnet, workers6):
    d = selection.get_variant("resnet-50", 10.0, resnet, snap(workers6), mode=0)
    assert d.variant is None and d.reject_kind == "no_variant"
    assert d.suggestion == "resnet-50__a30"          # closest latency (L8)


def test_active_variant_closest_to_slo_wins(resnet, workers6):
    s = snap(workers6, [("resnet-50__2080ti", "t0", A, 5), ("resnet-50__a30", "x0", A, 5)])
    d = selection.get_variant("resnet-50", 100.0, resnet, s, mode=0)
    assert (d.variant, d.worker, d.path) == ("resnet-50__2080ti", "t0", "active")
    # tight SLO: only a30 qualifies
    d = selection.get_variant("resnet-50", 26.0, resnet, s, mode=0)
    assert d.variant == "resnet-50__a30"


def test_overloaded_and_interfered_are_avoided(resnet, workers6):
    s = snap(workers6, [("resnet-50__2080ti", "t0", O, 40), ("resnet-50__a30", "x0", A, 5)])
    d = selection.get_variant("resnet-50", 100.0, resnet, s, mode=0)
    assert d.variant == "resnet-50__a30"
    s = snap(workers6, [("resnet-50__2080ti", "t0", I, 5), ("resnet-50__a30", "x0", A, 5)])
    assert selection.get_variant("resnet-50", 100.0, resnet, s, mode=0).variant == "resnet-50__a30"


def test_mode1_ignores_states(resnet, workers6):
    s = snap(workers6, [("resnet-50__2080ti", "t0", O, 40)])
    d = selection.get_variant("resnet-50", 100.0, resnet, s, mode=1)
    assert (d.variant, d.path) == ("resnet-50__2080ti", "active")


def test_request_bin_packing_fills_fullest_with_room(resnet, workers6):
    # sat=30 -> room below 30/1.05 = 28.57 qps
    s = snap(workers6, [("resnet-50__2080ti", "t0", A, 20), ("resnet-50__2080ti", "t1", A, 5)])
    assert selection.get_variant("resnet-50", 100.0, resnet, s, mode=0).worker == "t0"
    s = snap(workers6, [("resnet-50__2080ti", "t0", A, 29), ("resnet-50__2080ti", "t1", A, 5)])
    assert selection.get_variant("resnet-50", 100.0, resnet, s, mode=0).worker == "t1"
    # both past the threshold: least loaded
    s = snap(workers6, [("resnet-50__2080ti", "t0", A, 29.5), ("resnet-50__2080ti", "t1", A, 29)])
    assert selection.get_variant("resnet-50", 100.0, resnet, s, mode=0).worker == "t1"


def test_inactive_lowest_load_plus_inference(resnet, workers6):
    d = selection.get_variant("resnet-50", 100.0, resnet, snap(workers6), mode=0)
    # a30: 1200+25 is the lowest combined latency
    assert (d.variant, d.path) == ("resnet-50__a30", "inactive")
    assert d.worker in ("x0", "x1")


def test_placement_bin_packing_best_fit(resnet, workers6):
    ws = [w for w in workers6]
    ws[4].mem_free = 5 * GB     # x0 already holds others -> tighter fit
    d = selection.get_variant("resnet-50", 100.0, resnet, snap(ws), mode=0)
    assert d.worker == "x0"


def test_placement_skips_full_or_hot_workers(resnet, workers6):
    ws = list(workers6)
    ws[4].mem_free = int(1.2 * GB)       # below 1 GB slack + 0.5 GB instance
    ws[5].util = 95.0                    # above the 80% bin capacity
    d = selection.get_variant("resnet-50", 100.0, resnet, snap(ws), mode=0)
    # no a30 room (case d) -> next lowest load+inf: a5000 and 2080ti tie on load, a5000 faster
    assert d.variant == "resnet-50__a5000" and d.path == "inactive"


def test_join_loading_instance(resnet, workers6):
    s = snap(workers6, [("resnet-50__a30", "x1", L, 0)])
    d = selection.get_variant("resnet-50", 100.0, resnet, s, mode=0)
    assert (d.variant, d.worker, d.path) == ("resnet-50__a30", "x1", "loading")


def test_fallback_and_vm_flag_in_mode6(resnet):
    ws = [worker("t0", "2080ti", free=0)]      # nowhere to load anything
    s = snap(ws, [("resnet-50__2080ti", "t0", I, 10)])
    d = selection.get_variant("resnet-50", 100.0, resnet, s, mode=6)
    assert d.path == "fallback" and d.worker == "t0" and d.vm_scale_hw == "2080ti"


def test_no_capacity_reject(resnet):
    ws = [worker("t0", "2080ti", free=0)]
    d = selection.get_variant("resnet-50", 100.0, resnet, snap(ws), mode=0)
    assert d.variant is None and d.reject_kind == "no_capacity"


def test_blacklisted_worker_not_used(resnet, workers6):
    ws = list(workers6)
    ws[0].blacklisted = True
    s = snap(ws, [("resnet-50__2080ti", "t0", A, 1), ("resnet-50__a5000", "f0", A, 1)])
    d = selection.get_variant("resnet-50", 100.0, resnet, s, mode=0)
    assert d.variant == "resnet-50__a5000"
