"""Tests for src/credit_risk.py."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from credit_risk import (expected_loss, stressed_lgd, ifrs9_stage, provision,
                         top_reason_codes, adverse_impact_by_group, reason_codes_for_client)


def test_expected_loss():
    assert expected_loss(np.array([0.02]), 0.45, np.array([10000]))[0] == pytest.approx(90.0)


def test_stressed_lgd_cap():
    assert stressed_lgd(0.9, 2.0) == 1.0
    assert stressed_lgd(0.45) > 0.45


def test_ifrs9_stages():
    s = ifrs9_stage(np.array([0.01, 0.05, 0.30]), np.array([0.015, 0.20, 0.70]))
    assert s.tolist() == [1, 2, 2]


def test_provision_stage1_lower_than_stage2():
    p = provision(np.array([0.05, 0.05]), np.array([0.05, 0.30]), 0.45,
                  np.array([10000, 10000]), np.array([1, 2]))
    assert p[0] < p[1]


def test_reason_codes():
    r = top_reason_codes(None, ["a", "b", "c"], {"a": 10, "b": -5, "c": 2}, n=2)
    assert [x[0] for x in r] == ["b", "c"]


def test_adverse_impact_by_group():
    groups = np.array([1, 1, 2, 2])
    y = np.array([0, 1, 0, 0])
    approved = np.array([True, True, False, True])
    df = adverse_impact_by_group(groups, y, approved)
    assert len(df) == 2
    assert {"group", "approval_rate", "mean_cost"} <= set(df.columns)
    # group 1: one bad accepted -> mean 2.5 ; group 2: one good rejected -> mean 0.5
    g1 = df[df["group"] == 1].iloc[0]
    g2 = df[df["group"] == 2].iloc[0]
    assert g1["mean_cost"] > g2["mean_cost"]


def test_reason_codes_for_client_exact_bins():
    card = pd.DataFrame({
        "variable": ["a", "a", "b", "b"],
        "woe": [0.1, -0.2, 0.5, -0.3],
        "points": [10.0, -4.0, 20.0, -1.0],
    })
    client = pd.Series({"a": -0.2, "b": 0.5})
    codes = reason_codes_for_client(client, ["a", "b"], card, n=2)
    assert codes[0] == ("a", -4.0)
    assert codes[1] == ("b", 20.0)
