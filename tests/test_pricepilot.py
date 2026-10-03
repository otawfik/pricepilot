"""Tests for PricePilot: data, elasticity recovery, optimizer math, simulator."""

import numpy as np
import pandas as pd
import pytest

from pricepilot.data import generate_transactions
from pricepilot.elasticity import fit_elasticity
from pricepilot.optimize import (
    optimal_price, demand_at_price, profit_at_price, optimize_catalog, summarize,
)
from pricepilot.simulate import what_if


@pytest.fixture(scope="module")
def panel():
    df, true = generate_transactions(n_days=730, seed=7, return_true_params=True)
    return df, true


def test_generate_shape(panel):
    df, true = panel
    assert set(["sku_id", "price", "unit_cost", "promo", "units_sold"]) <= set(df.columns)
    assert df["sku_id"].nunique() == 60
    assert (df["price"] >= df["unit_cost"]).all()
    assert df["units_sold"].min() >= 0


def test_elasticities_negative(panel):
    df, _ = panel
    elas = fit_elasticity(df)
    assert (elas["elasticity"] < 0).mean() > 0.85


def test_elasticity_recovery(panel):
    """Estimated elasticities should be close to the known true values."""
    df, true = panel
    elas = fit_elasticity(df)
    merged = elas.merge(true, on="sku_id")
    err = (merged["elasticity"] - merged["true_elasticity"]).abs()
    assert (err < 0.5).mean() > 0.75, f"median err {err.median():.3f}"


def test_lerner_optimality():
    """Closed form p* = c*b/(1+b) must match numeric maximization."""
    from scipy.optimize import minimize_scalar
    c, b, p0, q0 = 6.0, -2.0, 10.0, 100.0
    p_star = optimal_price(c, b, p0, move_limit=10.0, margin_floor=0.0)
    assert p_star == pytest.approx(c * b / (1 + b), rel=1e-9)
    num = minimize_scalar(lambda p: -profit_at_price(p, c, q0, p0, b),
                          bounds=(c * 1.01, p0 * 5), method="bounded")
    assert p_star == pytest.approx(num.x, rel=1e-3)


def test_optimizer_uplift_nonnegative(panel):
    df, _ = panel
    elas = fit_elasticity(df)
    opt = optimize_catalog(df, elas)
    assert (opt["daily_uplift"] >= -1e-6).all()
    s = summarize(opt)
    assert s["uplift_pct"] >= 0


def test_bounds_and_margin(panel):
    """Guardrails hold on exact (unrounded) prices: never cut more than the
    move limit, never exceed it unless the margin floor forces it higher,
    and margin never drops below the floor."""
    df, _ = panel
    elas = fit_elasticity(df)
    for _, r in elas.iterrows():
        p0, c, b = r["current_price"], r["unit_cost"], r["elasticity"]
        p = optimal_price(c, b, p0, move_limit=0.2, margin_floor=0.2)
        lo, hi = p0 * 0.8, p0 * 1.2
        assert p >= lo - 1e-9, r["sku_id"]
        assert p <= hi + 1e-9 or c / 0.8 > hi - 1e-9, r["sku_id"]
        assert (p - c) / p >= 0.2 - 1e-9, r["sku_id"]


def test_whatif_monotonic(panel):
    df, _ = panel
    elas = fit_elasticity(df)
    row = elas[(elas["elasticity"] < -1.5)].iloc[0]
    low = what_if(row, row["current_price"] * 0.9)
    high = what_if(row, row["current_price"] * 1.1)
    assert low["predicted_daily_units"] > high["predicted_daily_units"]
    assert low["price_change_pct"] < 0 < high["price_change_pct"]


def test_inelastic_goes_to_upper_bound():
    p_star = optimal_price(unit_cost=5.0, elasticity=-0.5, current_price=10.0,
                           move_limit=0.3, margin_floor=0.0)
    assert p_star == pytest.approx(13.0)


def test_bad_estimate_holds_price():
    assert optimal_price(5.0, 0.4, 10.0) == 10.0
    assert optimal_price(5.0, float("nan"), 10.0) == 10.0


def test_demand_forecast_consistency():
    q = demand_at_price(100.0, 10.0, 10.0, -2.0)
    assert q == pytest.approx(100.0)
    q2 = demand_at_price(100.0, 10.0, 20.0, -2.0)
    assert q2 == pytest.approx(25.0)
