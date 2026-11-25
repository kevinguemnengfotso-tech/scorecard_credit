"""Tests for src/credit_risk.py."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from credit_risk import expected_loss, stressed_lgd, ifrs9_stage, provision, top_reason_codes


def test_expected_loss():
    assert expected_loss(np.array([0.02]), 0.45, np.array([10000]))[0] == pytest.approx(90.0)


def test_stressed_lgd_cap():
    assert stressed_lgd(0.9, 2.0) == 1.0
    assert stressed_lgd(0.45) > 0.45


def test_ifrs9_stages():
    s = ifrs9_stage(np.array([0.01, 0.05, 0.30]), np.array([0.015, 0.20, 0.60]))
    assert s.tolist() == [1, 2, 2]


def test_provision_stage1_lower_than_stage2():
    p = provision(np.array([0.05, 0.05]), np.array([0.05, 0.30]), 0.45,
                  np.array([10000, 10000]), np.array([1, 2]))
    assert p[0] < p[1]


def test_reason_codes():
    r = top_reason_codes(None, ["a", "b", "c"], {"a": 10, "b": -5, "c": 2}, n=2)
    assert [x[0] for x in r] == ["b", "c"]