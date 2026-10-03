"""Profit-optimal pricing from estimated elasticities.

Economics: with constant elasticity b < -1, profit pi(p) = (p - c) * A * p^b
is maximized at p* = c * b / (1 + b), i.e. the Lerner index (p* - c) / p* = -1/b.
We use the closed form, then clip to business guardrails (price-move bounds and
a margin floor). Inelastic SKUs (b in [-1, 0)) have profit increasing in price,
so they go to the upper bound; non-negative estimates are left untouched.
"""

import numpy as np
import pandas as pd


def optimal_price(unit_cost, elasticity, current_price,
                  move_limit=0.30, margin_floor=0.15):
    """Closed-form optimal price with guardrails."""
    if not np.isfinite(elasticity) or elasticity >= 0:
        return current_price  # bad estimate: do nothing
    if elasticity < -1:
        p_star = unit_cost * elasticity / (1 + elasticity)
    else:
        p_star = current_price * (1 + move_limit)  # inelastic: raise to bound
    lo = current_price * (1 - move_limit)
    hi = current_price * (1 + move_limit)
    p_star = min(max(p_star, lo), hi)
    min_price_for_margin = unit_cost / (1 - margin_floor)
    return max(p_star, min_price_for_margin)


def demand_at_price(avg_units, current_price, new_price, elasticity):
    """Constant-elasticity demand forecast: Q(p) = Q0 * (p / p0)^b."""
    return avg_units * (new_price / current_price) ** elasticity


def profit_at_price(price, unit_cost, avg_units, current_price, elasticity):
    q = demand_at_price(avg_units, current_price, price, elasticity)
    return (price - unit_cost) * q


def optimize_catalog(transactions, elas_df, move_limit=0.30, margin_floor=0.15):
    """Price every SKU for max profit. Returns per-SKU results + totals."""
    records = []
    for _, row in elas_df.iterrows():
        sku = row["sku_id"]
        p0 = row["current_price"]
        c = row["unit_cost"]
        b = row["elasticity"]
        q0 = row["avg_daily_units"]
        p_star = optimal_price(c, b, p0, move_limit, margin_floor)
        pi0 = profit_at_price(p0, c, q0, p0, b)
        pi1 = profit_at_price(p_star, c, q0, p0, b)
        records.append(
            {
                "sku_id": sku,
                "category": row["category"],
                "elasticity": b,
                "significant": bool(row["significant"]),
                "current_price": round(p0, 2),
                "optimal_price": round(p_star, 2),
                "price_change_pct": round(100 * (p_star - p0) / p0, 1),
                "current_daily_profit": round(pi0, 2),
                "optimal_daily_profit": round(pi1, 2),
                "daily_uplift": round(pi1 - pi0, 2),
            }
        )
    out = pd.DataFrame(records).sort_values("daily_uplift", ascending=False)
    return out.reset_index(drop=True)


def summarize(opt_df):
    """Aggregate business impact."""
    total_now = float(opt_df["current_daily_profit"].sum())
    total_opt = float(opt_df["optimal_daily_profit"].sum())
    uplift = total_opt - total_now
    return {
        "n_skus": len(opt_df),
        "daily_profit_now": round(total_now, 2),
        "daily_profit_opt": round(total_opt, 2),
        "daily_uplift": round(uplift, 2),
        "uplift_pct": round(100 * uplift / total_now, 2) if total_now else 0.0,
        "annualized_uplift": round(uplift * 365, 2),
        "skus_repriced": int((opt_df["price_change_pct"].abs() > 0.5).sum()),
    }
