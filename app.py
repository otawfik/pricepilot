"""PricePilot dashboard: summary, opportunities, elasticity chart, what-if tool."""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, render_template_string

from pricepilot.optimize import summarize
from pricepilot.simulate import what_if

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))


def load():
    opt = pd.read_csv(os.path.join(BASE, "data", "optimized_prices.csv"))
    elas = pd.read_csv(os.path.join(BASE, "data", "elasticities.csv"))
    return opt, elas


def chart_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


PAGE = """
<!doctype html><html><head><title>PricePilot</title>
<style>body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}
.cards{display:flex;gap:1rem;flex-wrap:wrap}.card{border:1px solid #ddd;border-radius:8px;padding:1rem;min-width:180px}
.card b{font-size:1.4rem}table{border-collapse:collapse;width:100%;margin-top:1rem}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:right}th{background:#f5f5f5}
td:first-child,th:first-child{text-align:left}img{max-width:100%}</style>
</head><body>
<h1>PricePilot: price optimization dashboard</h1>
<div class="cards">
<div class="card"><div>SKUs analyzed</div><b>{{s.n_skus}}</b></div>
<div class="card"><div>Daily profit now</div><b>${{"{:,.0f}".format(s.daily_profit_now)}}</b></div>
<div class="card"><div>Daily profit optimized</div><b>${{"{:,.0f}".format(s.daily_profit_opt)}}</b></div>
<div class="card"><div>Uplift</div><b style="color:green">+{{s.uplift_pct}}% (${{"{:,.0f}".format(s.annualized_uplift)}}/yr)</b></div>
</div>
<h2>Top repricing opportunities</h2>
{{table|safe}}
<h2>Elasticity distribution</h2>
<img src="data:image/png;base64,{{hist}}">
<h2>What-if simulator</h2>
<form method="post" action="/whatif">
SKU <input name="sku" value="SKU-0000" size="10">
new price <input name="price" value="10.00" size="8">
<input type="submit" value="Simulate"></form>
{% if result %}<h3>Result</h3><table>
{% for k,v in result.items() %}<tr><td>{{k}}</td><td>{{v}}</td></tr>{% endfor %}
</table>{% endif %}
</body></html>
"""


@app.route("/", methods=["GET"])
def index():
    return _render(None)


@app.route("/whatif", methods=["POST"])
def whatif():
    opt, elas = load()
    row = elas[elas["sku_id"] == request.form["sku"]]
    if row.empty:
        return _render({"error": "unknown SKU"})
    return _render(what_if(row.iloc[0], float(request.form["price"])))


def _render(result):
    opt, elas = load()
    s = summarize(opt)
    top = opt.head(15)[["sku_id", "category", "elasticity", "current_price",
                        "optimal_price", "price_change_pct", "daily_uplift"]]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(elas["elasticity"], bins=20, edgecolor="white")
    ax.axvline(-1, color="red", linestyle="--", label="unit elasticity")
    ax.set_xlabel("price elasticity of demand")
    ax.set_ylabel("SKUs")
    ax.legend()
    return render_template_string(PAGE, s=s, table=top.to_html(index=False),
                                  hist=chart_png(fig), result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
