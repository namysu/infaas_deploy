"""State machine (Fig. 4) and the ILP objective (§4.2.1)."""
import pytest
from conftest import GB, prof

from infaas.common import config
from infaas.policy import costs, states


def test_overloaded_at_peak():
    assert states.next_state(states.ACTIVE, 30, 40, 40, 30) == states.OVERLOADED
    assert states.next_state(states.OVERLOADED, 29, 40, 40, 30) == states.ACTIVE


def test_interfered_needs_latency_and_load():
    # 1.5x factor for profiled >= 10 ms, qps above 0.3 x capacity
    assert states.next_state(states.ACTIVE, 20, 61, 40, 30) == states.INTERFERED
    assert states.next_state(states.ACTIVE, 5, 61, 40, 30) == states.ACTIVE     # low load
    assert states.next_state(states.ACTIVE, 20, 59, 40, 30) == states.ACTIVE


def test_interfered_hysteresis():
    assert states.next_state(states.INTERFERED, 20, 55, 40, 30) == states.INTERFERED  # >= 1.25x
    assert states.next_state(states.INTERFERED, 20, 49, 40, 30) == states.ACTIVE


def test_small_latency_factor():
    assert states.interfered_factor(5.0) == pytest.approx(3.0)      # min(5, 15/5)
    assert states.interfered_factor(2.0) == pytest.approx(5.0)


def test_cost_function():
    p = prof("m", "a30", 10, load=2000, mem=GB)
    c = config.HW_COST["a30"] * GB / 1e9
    assert costs.instance_cost(p) == pytest.approx(c)
    assert costs.action_cost(p, 2) == pytest.approx(c * (2 + config.LAMBDA * 2.0 * 2))
    assert costs.action_cost(p, -1) == pytest.approx(-c)


def test_instances_needed_includes_slack():
    assert costs.instances_needed(0, 10) == 0
    assert costs.instances_needed(10, 10) == 2      # 10 x 1.05 > 10
    assert costs.instances_needed(9.5, 10) == 1
