"""Demand elasticity estimation via log-log regression.

For each SKU we fit:
    ln(Q_it) = a_i + b_i * ln(P_it) + g_i * promo_it + t_i * ln(Pcomp_it)
               + dow effects + trend + e_it

The coefficient b_i on log price IS the price elasticity of demand.
Standard errors come from the OLS variance formula; p-values from the
t-distribution (scipy).
"""

import numpy as np
import pandas as pd
from scipy import stats


def _design_matrix(group):
    g = group.sort_values("date").reset_index(drop=True)
    # drop zero-sale days for the log transform (rare)
    g = g[g["units_sold"] > 0].copy()
    n = len(g)
    if n < 60:
        return None, None
    log_q = np.log(g["units_sold"].values)
    log_p = np.log(g["price"].values)
    promo = g["promo"].values.astype(float)
    log_pc = np.log(g["competitor_price"].values)
    dow = pd.get_dummies(g["date"].dt.dayofweek, drop_first=True).values.astype(float)
    trend = np.arange(n, dtype=float) / n  # 0..1
    X = np.column_stack([np.ones(n), log_p, promo, log_pc, dow, trend])
    return X, log_q


def fit_elasticity(df):
    """Fit per-SKU elasticities. Returns a DataFrame, one row per SKU."""
    records = []
    for sku_id, group in df.groupby("sku_id"):
        X, y = _design_matrix(group)
        if X is None:
            continue
        coef, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        n, k = X.shape
        dof = n - k
        resid = y - X @ coef
        s2 = float(resid @ resid / dof)
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        se = float(np.sqrt(s2 * xtx_inv[1, 1]))
        elasticity = float(coef[1])
        t_stat = elasticity / se if se > 0 else 0.0
        p_value = float(2 * stats.t.sf(abs(t_stat), dof))
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1 - (resid @ resid) / ss_tot) if ss_tot > 0 else 0.0
        records.append(
            {
                "sku_id": sku_id,
                "category": group["category"].iloc[0],
                "elasticity": round(elasticity, 3),
                "std_err": round(se, 4),
                "p_value": p_value,
                "r_squared": round(r2, 3),
                "n_obs": n,
                "current_price": float(group.sort_values("date")["price"].iloc[-1]),
                "unit_cost": float(group["unit_cost"].iloc[0]),
                "avg_daily_units": float(group["units_sold"].mean()),
            }
        )
    out = pd.DataFrame(records)
    out["significant"] = (out["p_value"] < 0.05) & (out["elasticity"] < 0)
    return out.sort_values("sku_id").reset_index(drop=True)
