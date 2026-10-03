"""What-if scenario simulator: price change -> demand/profit impact."""

from .optimize import demand_at_price, profit_at_price


def what_if(elas_row, new_price):
    """Simulate moving one SKU to new_price. elas_row is a row of the
    elasticity DataFrame (dict-like with sku_id, current_price, unit_cost,
    avg_daily_units, elasticity)."""
    p0 = float(elas_row["current_price"])
    c = float(elas_row["unit_cost"])
    q0 = float(elas_row["avg_daily_units"])
    b = float(elas_row["elasticity"])
    q_new = demand_at_price(q0, p0, new_price, b)
    pi_old = profit_at_price(p0, c, q0, p0, b)
    pi_new = profit_at_price(new_price, c, q0, p0, b)
    return {
        "sku_id": elas_row["sku_id"],
        "old_price": round(p0, 2),
        "new_price": round(new_price, 2),
        "price_change_pct": round(100 * (new_price - p0) / p0, 1),
        "predicted_daily_units": round(q_new, 1),
        "unit_change_pct": round(100 * (q_new - q0) / q0, 1) if q0 else 0.0,
        "old_daily_profit": round(pi_old, 2),
        "new_daily_profit": round(pi_new, 2),
        "profit_change_pct": round(100 * (pi_new - pi_old) / pi_old, 1) if pi_old else 0.0,
    }
