"""PricePilot CLI: generate data, fit elasticities, optimize prices, run what-ifs."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from pricepilot.data import generate_transactions
from pricepilot.elasticity import fit_elasticity
from pricepilot.optimize import optimize_catalog, summarize
from pricepilot.simulate import what_if

DATA = "data/transactions.csv"
ELAS = "data/elasticities.csv"
OPT = "data/optimized_prices.csv"


def cmd_generate(args):
    df, params = generate_transactions(n_days=args.days, seed=args.seed,
                                       return_true_params=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA, index=False)
    params.to_csv("data/true_params.csv", index=False)
    print(f"generated {len(df):,} rows across {df['sku_id'].nunique()} SKUs -> {DATA}")


def cmd_fit(args):
    df = pd.read_csv(DATA, parse_dates=["date"])
    elas = fit_elasticity(df)
    elas.to_csv(ELAS, index=False)
    sig = elas["significant"].sum()
    print(f"fit {len(elas)} SKUs, {sig} with significant negative elasticity")
    print(elas[["sku_id", "category", "elasticity", "p_value", "r_squared"]]
          .head(10).to_string(index=False))


def cmd_optimize(args):
    elas = pd.read_csv(ELAS)
    opt = optimize_catalog(pd.read_csv(DATA, parse_dates=["date"]), elas,
                           move_limit=args.move_limit, margin_floor=args.margin_floor)
    opt.to_csv(OPT, index=False)
    s = summarize(opt)
    print(json.dumps(s, indent=2))
    print("\ntop 10 repricing opportunities:")
    print(opt[["sku_id", "category", "current_price", "optimal_price",
               "price_change_pct", "daily_uplift"]].head(10).to_string(index=False))


def cmd_whatif(args):
    elas = pd.read_csv(ELAS)
    row = elas[elas["sku_id"] == args.sku]
    if row.empty:
        print(f"unknown SKU: {args.sku}")
        sys.exit(1)
    print(json.dumps(what_if(row.iloc[0], args.price), indent=2))


def main():
    ap = argparse.ArgumentParser(prog="pricepilot",
                                 description="Retail price optimization engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate synthetic transactions")
    g.add_argument("--days", type=int, default=730)
    g.add_argument("--seed", type=int, default=42)

    sub.add_parser("fit", help="estimate demand elasticities")

    o = sub.add_parser("optimize", help="compute profit-optimal prices")
    o.add_argument("--move-limit", type=float, default=0.30)
    o.add_argument("--margin-floor", type=float, default=0.15)

    w = sub.add_parser("whatif", help="simulate a price change")
    w.add_argument("--sku", required=True)
    w.add_argument("--price", type=float, required=True)

    args = ap.parse_args()
    {"generate": cmd_generate, "fit": cmd_fit,
     "optimize": cmd_optimize, "whatif": cmd_whatif}[args.cmd](args)


if __name__ == "__main__":
    main()
