"""Model-Autoscaler greedy (§4.2.2) and VM-Autoscaler rules (§4.2.3)."""
from conftest import GB, prof

from infaas.policy import scaling
from infaas.policy.scaling import ClusterStat, LocalStat, WorkerStat


def _by_model(ps):
    return {ps[0].model: ps}


def test_no_scale_with_headroom(resnet):
    p = {v.variant: v for v in resnet}
    local = {"resnet-50__2080ti": LocalStat(20, 100)}
    assert scaling.scale_up_options(local, p, _by_model(resnet),
                                    {"resnet-50__2080ti": ClusterStat(1, 20)}) == {}


def test_scale_up_offers_replicate_and_upgrade_sorted_by_cost(resnet):
    p = {v.variant: v for v in resnet}
    local = {"resnet-50__2080ti": LocalStat(40, 100)}
    ups = scaling.scale_up_options(local, p, _by_model(resnet),
                                   {"resnet-50__2080ti": ClusterStat(1, 40)})
    opts = ups["resnet-50__2080ti"]
    kinds = {(o.kind, o.dst_variant) for o in opts}
    assert ("replicate", "resnet-50__2080ti") in kinds
    assert ("upgrade", "resnet-50__a30") in kinds
    assert opts == sorted(opts, key=lambda o: (o.cost, o.dst_variant))
    # replicate: need ceil(40*1.05/30)=2 instances -> +1 at cost 1x(1+1.5)
    rep = next(o for o in opts if o.kind == "replicate")
    assert rep.count == 1


def test_upgrade_respects_min_slo():
    ps = [prof("m", "2080ti", 40, sat=30), prof("m", "a30", 60, sat=100)]   # faster but slower lat
    p = {v.variant: v for v in ps}
    ups = scaling.scale_up_options({"m__2080ti": LocalStat(40, 50)}, p, _by_model(ps),
                                   {"m__2080ti": ClusterStat(1, 40)})
    assert all(o.kind != "upgrade" for o in ups["m__2080ti"])


def test_spare_instances_elsewhere_means_no_request(resnet):
    p = {v.variant: v for v in resnet}
    local = {"resnet-50__2080ti": LocalStat(31, 100)}          # this one is full ...
    ups = scaling.scale_up_options(local, p, _by_model(resnet),
                                   {"resnet-50__2080ti": ClusterStat(2, 33)})  # ... the other idle
    assert ups == {}


def test_scale_down_remove_idle_replica(resnet):
    a = resnet[0]
    opt = scaling.scale_down_option(a.variant, LocalStat(0, 100), a, _by_model(resnet),
                                    ClusterStat(2, 10))
    assert opt.kind == "remove"


def test_scale_down_last_instance_only_at_zero(resnet):
    a = resnet[0]
    assert scaling.scale_down_option(a.variant, LocalStat(1, 100), a, _by_model(resnet),
                                     ClusterStat(1, 1)) is None
    assert scaling.scale_down_option(a.variant, LocalStat(0, None), a, _by_model(resnet),
                                     ClusterStat(1, 0)).kind == "remove"


def test_downgrade_only_when_cost_drops():
    # same memory: a30 (4) -> 2080ti (1) costs 1*(1+λ·T) - 4 < 0 for T < 3 s
    ps = [prof("m", "2080ti", 40, load=1000, sat=30, mem=GB),
          prof("m", "a5000", 30, load=1000, sat=40, mem=GB),
          prof("m", "a30", 25, load=1000, sat=50, mem=GB)]
    top = ps[2]
    opt = scaling.scale_down_option(top.variant, LocalStat(5, 100), top, _by_model(ps),
                                    ClusterStat(1, 5))
    assert opt.kind == "downgrade" and opt.dst_variant == "m__2080ti" and opt.cost < 0
    # a slow load makes the move not worth it
    ps[0] = prof("m", "2080ti", 40, load=4000, sat=30, mem=GB)
    assert scaling.scale_down_option(top.variant, LocalStat(5, 100), top, _by_model(ps),
                                     ClusterStat(1, 5)) is None
    # SLO too tight for the cheaper variant
    ps[0] = prof("m", "2080ti", 40, load=1000, sat=30, mem=GB)
    assert scaling.scale_down_option(top.variant, LocalStat(5, 35), top, _by_model(ps),
                                     ClusterStat(1, 5)) is None


def test_scale_down_timer_waits_t_v():
    t = scaling.ScaleDownTimer()
    t.observe("v", True, 100.0)
    assert not t.ready("v", 101.0, 2.3)
    assert t.ready("v", 103.0, 2.3)          # ceil(2.3) = 3 slots of 1 s
    t.observe("v", False, 103.5)
    assert not t.ready("v", 104.0, 2.3)


def _w(name, hw, util=10, inter=False, over=False):
    return WorkerStat(name, hw, util, 5, inter, over, True)


def test_vm_rule1_util():
    ws = [_w("a", "a30", 90), _w("b", "a30", 85), _w("c", "2080ti", 10)]
    assert scaling.vm_scale_up(ws, [], {"a30": 3, "2080ti": 2}) == ("a30", "rule1_util")
    assert scaling.vm_scale_up(ws, [], {"a30": 2, "2080ti": 2}) is None   # at max


def test_vm_rule2_interfered():
    ws = [_w("a", "a5000", inter=True), _w("b", "a5000", inter=True)]
    assert scaling.vm_scale_up(ws, [], {"a5000": 3}) == ("a5000", "rule2_interfered")


def test_vm_rule3_overloaded_prefers_most_overloaded_type():
    ws = [_w("a", "2080ti", over=True), _w("b", "2080ti", over=True),
          _w("c", "a30", over=True), _w("d", "a5000", over=True), _w("e", "a5000", over=False)]
    # exactly 80% is not "more than 80%"
    assert scaling.vm_scale_up(ws, [], {"2080ti": 3, "a30": 2, "a5000": 3}) is None
    ws[4] = _w("e", "a5000", over=True)
    # 2080ti and a5000 tie on count -> the cheaper type
    assert scaling.vm_scale_up(ws, [], {"2080ti": 3, "a30": 2, "a5000": 3}) == \
        ("2080ti", "rule3_overloaded")
    # 2080ti at max -> the other most-overloaded type
    assert scaling.vm_scale_up(ws, [], {"2080ti": 2, "a30": 2, "a5000": 3}) == \
        ("a5000", "rule3_overloaded")


def test_vm_flag():
    ws = [_w("a", "a30")]
    assert scaling.vm_scale_up(ws, ["a30"], {"a30": 2}) == ("a30", "vm_scale_flag")


def test_vm_scale_down_condition():
    assert scaling.vm_scale_down_ok([_w("a", "a30", util=3), _w("b", "a30", util=50)])
    assert not scaling.vm_scale_down_ok([_w("a", "a30", util=30), _w("b", "a30", util=50)])
