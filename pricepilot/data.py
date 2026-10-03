"""Synthetic retail transaction generator.

Produces a realistic SKU-day panel: prices move with promos and a random walk,
competitor prices drift nearby, and demand follows a constant-elasticity curve
with seasonality, promo lift, and noise. True elasticities are known, so the
estimation module can be validated against ground truth.
"""

import numpy as np
import pandas as pd

# category -> (number of SKUs, mean true price elasticity)
CATEGORIES = {
    "Beverages": (10, -2.2),
    "Snacks": (10, -1.8),
    "Pantry": (10, -1.4),
    "Frozen": (10, -2.6),
    "Household": (10, -1.2),
    "Personal Care": (10, -1.6),
}


def generate_transactions(n_days=730, seed=42, return_true_params=False):
    """Generate a SKU-day transaction panel.

    Returns a DataFrame with columns:
      date, sku_id, category, price, unit_cost, competitor_price, promo, units_sold
    If return_true_params is True, also returns a DataFrame of true parameters
    per SKU (true_elasticity, base_price, base_demand).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    dow = dates.dayofweek.values
    # weekly seasonality (weekend lift) + yearly seasonality
    weekly = 1.0 + 0.18 * (dow >= 5).astype(float)
    yearly = 1.0 + 0.12 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)

    rows = []
    true_params = []
    sku_counter = 0
    for category, (n_skus, elas_mean) in CATEGORIES.items():
        for _ in range(n_skus):
            sku_id = f"SKU-{sku_counter:04d}"
            sku_counter += 1
            base_price = float(rng.lognormal(mean=2.2, sigma=0.6))  # ~$9 avg
            base_price = round(max(base_price, 1.5), 2)
            margin = float(rng.uniform(0.30, 0.60))
            unit_cost = round(base_price * (1 - margin), 2)
            true_elas = float(rng.normal(elas_mean, 0.35))
            true_elas = min(max(true_elas, -3.8), -1.05)  # elastic, negative
            cross_elas = float(rng.uniform(0.25, 0.75))  # substitution w/ competitor
            base_demand = float(rng.lognormal(mean=3.4, sigma=0.7))  # ~30 units/day

            # price path: slow random walk + promo dips + permanent price tests
            walk = np.cumsum(rng.normal(0, 0.006, n_days))
            price_mult = np.exp(walk - walk.mean())
            # price tests: every ~45-75 days a lasting +/-5-20% shift, giving
            # the regression clean identifying variation (as real retailers do)
            t = 0
            while True:
                t += int(rng.integers(45, 75))
                if t >= n_days:
                    break
                shock = float(rng.uniform(0.05, 0.20) * rng.choice([-1.0, 1.0]))
                price_mult[t:] *= (1 + shock)
            promo = np.zeros(n_days, dtype=int)
            # promos: random 7-day windows, ~8% of days
            n_promos = max(1, n_days // 90)
            for _ in range(n_promos):
                start = rng.integers(0, n_days - 7)
                depth = rng.uniform(0.15, 0.30)
                promo[start:start + 7] = 1
                price_mult[start:start + 7] *= (1 - depth)
            price = np.round(base_price * price_mult, 2)
            price = np.maximum(price, unit_cost * 1.05)  # never sell at a loss

            # competitor price: correlated drift around our base price
            comp = base_price * np.exp(np.cumsum(rng.normal(0, 0.006, n_days)))
            comp = np.round(comp, 2)

            # demand: constant elasticity + promo lift + seasonality + noise
            log_q = (
                np.log(base_demand)
                + true_elas * np.log(price / base_price)
                + cross_elas * np.log(comp / base_price)
                + 0.35 * promo
                + np.log(weekly * yearly)
                + rng.normal(0, 0.18, n_days)
            )
            units = np.random.default_rng(seed + sku_counter).poisson(
                np.exp(log_q)
            ).astype(float)
            # extra overdispersion noise
            units = np.maximum(units * rng.lognormal(0, 0.08, n_days), 0).round().astype(int)

            for i in range(n_days):
                rows.append(
                    {
                        "date": dates[i],
                        "sku_id": sku_id,
                        "category": category,
                        "price": price[i],
                        "unit_cost": unit_cost,
                        "competitor_price": comp[i],
                        "promo": promo[i],
                        "units_sold": units[i],
                    }
                )
            true_params.append(
                {
                    "sku_id": sku_id,
                    "category": category,
                    "true_elasticity": round(true_elas, 3),
                    "base_price": base_price,
                    "unit_cost": unit_cost,
                    "base_demand": round(base_demand, 1),
                }
            )

    df = pd.DataFrame(rows)
    if return_true_params:
        return df, pd.DataFrame(true_params)
    return df


def main():
    import os

    df, params = generate_transactions(return_true_params=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    params.to_csv("data/true_params.csv", index=False)
    print(f"wrote data/transactions.csv ({len(df):,} rows, {df['sku_id'].nunique()} SKUs)")
    print(f"wrote data/true_params.csv")


if __name__ == "__main__":
    main()
