"""
solver.py — Production Planning Optimization Engine
Business Problem: Multi-Product, Multi-Machine Manufacturing
Author: Kartikay Verma
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
#  BUSINESS DATA
# ──────────────────────────────────────────────────────────────
PRODUCTS = ["Laptop", "Tablet", "Smartphone", "Smartwatch", "Headphones"]

# Profit per unit (₹)
PROFIT = {
    "Laptop":     15_000,
    "Tablet":      8_000,
    "Smartphone": 12_000,
    "Smartwatch":  5_500,
    "Headphones":  3_200,
}

# Max market demand per week (units)
DEMAND = {
    "Laptop":     200,
    "Tablet":     350,
    "Smartphone": 500,
    "Smartwatch": 400,
    "Headphones": 600,
}

# Minimum contractual production per week (units)
MIN_PROD = {
    "Laptop":      20,
    "Tablet":      30,
    "Smartphone":  50,
    "Smartwatch":  25,
    "Headphones":  40,
}

# Machine hours required per unit {product: {machine: hours}}
MACHINES = ["Assembly", "Testing", "Packaging"]
MACHINE_HOURS = {
    "Laptop":     {"Assembly": 4.5, "Testing": 2.0, "Packaging": 0.8},
    "Tablet":     {"Assembly": 2.0, "Testing": 1.5, "Packaging": 0.5},
    "Smartphone": {"Assembly": 1.8, "Testing": 1.2, "Packaging": 0.4},
    "Smartwatch": {"Assembly": 1.0, "Testing": 0.8, "Packaging": 0.3},
    "Headphones": {"Assembly": 0.6, "Testing": 0.4, "Packaging": 0.2},
}

# Available machine hours per week
MACHINE_CAPACITY = {
    "Assembly":  1_800,
    "Testing":   1_200,
    "Packaging":   600,
}

# Raw material (kg) per unit
MATERIALS = ["Semiconductors", "Plastics", "Metals"]
MATERIAL_USE = {
    "Laptop":     {"Semiconductors": 0.5, "Plastics": 1.2, "Metals": 0.8},
    "Tablet":     {"Semiconductors": 0.3, "Plastics": 0.8, "Metals": 0.3},
    "Smartphone": {"Semiconductors": 0.4, "Plastics": 0.5, "Metals": 0.2},
    "Smartwatch": {"Semiconductors": 0.2, "Plastics": 0.2, "Metals": 0.1},
    "Headphones": {"Semiconductors": 0.1, "Plastics": 0.3, "Metals": 0.1},
}

# Available raw materials per week (kg)
MATERIAL_SUPPLY = {
    "Semiconductors": 300,
    "Plastics":       600,
    "Metals":         400,
}

# Budget constraint: variable cost per unit (₹)
VARIABLE_COST = {
    "Laptop":      6_000,
    "Tablet":      3_200,
    "Smartphone":  4_500,
    "Smartwatch":  2_000,
    "Headphones":    800,
}

WEEKLY_BUDGET = 5_000_000  # ₹50 lakh per week


# ──────────────────────────────────────────────────────────────
#  MODEL 1 — PROFIT MAXIMIZATION (LP)
# ──────────────────────────────────────────────────────────────
def solve_lp():
    prob = pulp.LpProblem("Production_Planning_Max_Profit", pulp.LpMaximize)

    x = {p: pulp.LpVariable(f"x_{p}", lowBound=0, cat="Continuous") for p in PRODUCTS}

    # Objective
    prob += pulp.lpSum(PROFIT[p] * x[p] for p in PRODUCTS), "Total_Profit"

    # Demand upper bounds
    for p in PRODUCTS:
        prob += x[p] <= DEMAND[p], f"Demand_{p}"

    # Minimum production (contractual)
    for p in PRODUCTS:
        prob += x[p] >= MIN_PROD[p], f"MinProd_{p}"

    # Machine hour constraints
    for m in MACHINES:
        prob += (pulp.lpSum(MACHINE_HOURS[p][m] * x[p] for p in PRODUCTS)
                 <= MACHINE_CAPACITY[m]), f"Machine_{m}"

    # Material supply constraints
    for mat in MATERIALS:
        prob += (pulp.lpSum(MATERIAL_USE[p][mat] * x[p] for p in PRODUCTS)
                 <= MATERIAL_SUPPLY[mat]), f"Material_{mat}"

    # Budget constraint
    prob += pulp.lpSum(VARIABLE_COST[p] * x[p] for p in PRODUCTS) <= WEEKLY_BUDGET, "Budget"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    results = {
        "status": pulp.LpStatus[prob.status],
        "total_profit": pulp.value(prob.objective),
        "units": {p: round(pulp.value(x[p]), 2) for p in PRODUCTS},
        "revenue": {p: round(PROFIT[p] * pulp.value(x[p]), 2) for p in PRODUCTS},
        "cost":    {p: round(VARIABLE_COST[p] * pulp.value(x[p]), 2) for p in PRODUCTS},
        "machine_usage": {
            m: round(sum(MACHINE_HOURS[p][m] * pulp.value(x[p]) for p in PRODUCTS), 2)
            for m in MACHINES
        },
        "material_usage": {
            mat: round(sum(MATERIAL_USE[p][mat] * pulp.value(x[p]) for p in PRODUCTS), 2)
            for mat in MATERIALS
        },
        "shadow_prices": {c.name: round(c.pi, 4) for c in prob.constraints.values()},
        "slack": {c.name: round(c.slack, 4) for c in prob.constraints.values()},
    }
    results["total_cost"] = sum(results["cost"].values())
    results["total_revenue"] = sum(results["revenue"].values())
    return results, prob


# ──────────────────────────────────────────────────────────────
#  MODEL 2 — INTEGER LP (ILP) — units must be whole numbers
# ──────────────────────────────────────────────────────────────
def solve_ilp():
    prob = pulp.LpProblem("Production_Planning_ILP", pulp.LpMaximize)

    x = {p: pulp.LpVariable(f"x_{p}", lowBound=0, cat="Integer") for p in PRODUCTS}

    prob += pulp.lpSum(PROFIT[p] * x[p] for p in PRODUCTS)

    for p in PRODUCTS:
        prob += x[p] <= DEMAND[p]
        prob += x[p] >= MIN_PROD[p]

    for m in MACHINES:
        prob += pulp.lpSum(MACHINE_HOURS[p][m] * x[p] for p in PRODUCTS) <= MACHINE_CAPACITY[m]

    for mat in MATERIALS:
        prob += pulp.lpSum(MATERIAL_USE[p][mat] * x[p] for p in PRODUCTS) <= MATERIAL_SUPPLY[mat]

    prob += pulp.lpSum(VARIABLE_COST[p] * x[p] for p in PRODUCTS) <= WEEKLY_BUDGET

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return {
        "status": pulp.LpStatus[prob.status],
        "total_profit": pulp.value(prob.objective),
        "units": {p: int(pulp.value(x[p])) for p in PRODUCTS},
    }


# ──────────────────────────────────────────────────────────────
#  MODEL 3 — SENSITIVITY: what if budget increases by 10 L each step?
# ──────────────────────────────────────────────────────────────
def sensitivity_budget():
    budgets = list(range(3_000_000, 8_000_001, 500_000))
    profits = []
    for budget in budgets:
        prob = pulp.LpProblem("sens", pulp.LpMaximize)
        x = {p: pulp.LpVariable(f"x_{p}", lowBound=0, cat="Continuous") for p in PRODUCTS}
        prob += pulp.lpSum(PROFIT[p] * x[p] for p in PRODUCTS)
        for p in PRODUCTS:
            prob += x[p] <= DEMAND[p]
            prob += x[p] >= MIN_PROD[p]
        for m in MACHINES:
            prob += pulp.lpSum(MACHINE_HOURS[p][m] * x[p] for p in PRODUCTS) <= MACHINE_CAPACITY[m]
        for mat in MATERIALS:
            prob += pulp.lpSum(MATERIAL_USE[p][mat] * x[p] for p in PRODUCTS) <= MATERIAL_SUPPLY[mat]
        prob += pulp.lpSum(VARIABLE_COST[p] * x[p] for p in PRODUCTS) <= budget
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        profits.append(pulp.value(prob.objective) or 0)
    return budgets, profits


# ──────────────────────────────────────────────────────────────
#  MODEL 4 — WHAT-IF: Semiconductor supply shock (-20%)
# ──────────────────────────────────────────────────────────────
def solve_supply_shock():
    shocked = {**MATERIAL_SUPPLY, "Semiconductors": int(MATERIAL_SUPPLY["Semiconductors"] * 0.80)}
    prob = pulp.LpProblem("SupplyShock", pulp.LpMaximize)
    x = {p: pulp.LpVariable(f"x_{p}", lowBound=0, cat="Continuous") for p in PRODUCTS}
    prob += pulp.lpSum(PROFIT[p] * x[p] for p in PRODUCTS)
    for p in PRODUCTS:
        prob += x[p] <= DEMAND[p]
        prob += x[p] >= MIN_PROD[p]
    for m in MACHINES:
        prob += pulp.lpSum(MACHINE_HOURS[p][m] * x[p] for p in PRODUCTS) <= MACHINE_CAPACITY[m]
    for mat in MATERIALS:
        prob += pulp.lpSum(MATERIAL_USE[p][mat] * x[p] for p in PRODUCTS) <= shocked[mat]
    prob += pulp.lpSum(VARIABLE_COST[p] * x[p] for p in PRODUCTS) <= WEEKLY_BUDGET
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return {
        "status": pulp.LpStatus[prob.status],
        "total_profit": pulp.value(prob.objective),
        "units": {p: round(pulp.value(x[p]), 2) for p in PRODUCTS},
    }


# ──────────────────────────────────────────────────────────────
#  PLOTTING
# ──────────────────────────────────────────────────────────────
COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
GREY   = "#94A3B8"

def plot_all(lp, ilp, shock, budgets, profits):
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#F8FAFC")
    gs  = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.38)

    # ── 1. Optimal production units (LP vs ILP) ──
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(PRODUCTS))
    w = 0.35
    ax1.bar(x_pos - w/2, [lp["units"][p]  for p in PRODUCTS], w, label="LP (continuous)", color=COLORS[0], alpha=0.85)
    ax1.bar(x_pos + w/2, [ilp["units"][p] for p in PRODUCTS], w, label="ILP (integer)",   color=COLORS[1], alpha=0.85)
    ax1.bar(x_pos, [DEMAND[p] for p in PRODUCTS], 0.01, color="black", alpha=0.3, label="Max Demand")
    ax1.set_xticks(x_pos); ax1.set_xticklabels(PRODUCTS, fontsize=9)
    ax1.set_ylabel("Units/week"); ax1.set_title("1. Optimal Production Plan\n(LP vs ILP)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_facecolor("#F8FAFC")

    # ── 2. Revenue & Profit per product ──
    ax2 = fig.add_subplot(gs[0, 1])
    rev  = [lp["revenue"][p]/1e6 for p in PRODUCTS]
    cost = [lp["cost"][p]/1e6    for p in PRODUCTS]
    prft = [(lp["revenue"][p] - lp["cost"][p])/1e6 for p in PRODUCTS]
    ax2.bar(PRODUCTS, rev,  color=COLORS[0], alpha=0.5, label="Revenue (₹M)")
    ax2.bar(PRODUCTS, cost, color=COLORS[3], alpha=0.7, label="Cost (₹M)")
    ax2.plot(PRODUCTS, prft, "D-", color=COLORS[2], ms=7, lw=2, label="Profit (₹M)")
    ax2.set_ylabel("₹ Millions"); ax2.set_title("2. Revenue, Cost & Profit\nby Product", fontweight="bold")
    ax2.legend(fontsize=8); ax2.set_facecolor("#F8FAFC")

    # ── 3. Machine utilisation ──
    ax3 = fig.add_subplot(gs[1, 0])
    util_pct = {m: lp["machine_usage"][m] / MACHINE_CAPACITY[m] * 100 for m in MACHINES}
    bars = ax3.barh(MACHINES, list(util_pct.values()),
                    color=[COLORS[3] if v > 90 else COLORS[0] for v in util_pct.values()],
                    alpha=0.85)
    ax3.axvline(100, color="red", lw=1.5, ls="--", label="Capacity (100%)")
    ax3.axvline(80,  color="orange", lw=1.2, ls=":", label="Warning (80%)")
    for bar, val in zip(bars, util_pct.values()):
        ax3.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax3.set_xlim(0, 115); ax3.set_xlabel("Utilisation %")
    ax3.set_title("3. Machine Utilisation", fontweight="bold")
    ax3.legend(fontsize=8); ax3.set_facecolor("#F8FAFC")

    # ── 4. Material utilisation ──
    ax4 = fig.add_subplot(gs[1, 1])
    mat_pct = {m: lp["material_usage"][m] / MATERIAL_SUPPLY[m] * 100 for m in MATERIALS}
    bars4 = ax4.barh(MATERIALS, list(mat_pct.values()),
                     color=[COLORS[3] if v > 90 else COLORS[1] for v in mat_pct.values()],
                     alpha=0.85)
    ax4.axvline(100, color="red", lw=1.5, ls="--")
    for bar, val in zip(bars4, mat_pct.values()):
        ax4.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax4.set_xlim(0, 115); ax4.set_xlabel("Utilisation %")
    ax4.set_title("4. Raw Material Utilisation", fontweight="bold")
    ax4.set_facecolor("#F8FAFC")

    # ── 5. Sensitivity: profit vs budget ──
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot([b/1e6 for b in budgets], [p/1e6 for p in profits], "o-",
             color=COLORS[0], lw=2.5, ms=7)
    ax5.axvline(WEEKLY_BUDGET/1e6, color=COLORS[3], ls="--", lw=2, label=f"Current Budget (₹{WEEKLY_BUDGET/1e6:.1f}M)")
    ax5.fill_between([b/1e6 for b in budgets], [p/1e6 for p in profits], alpha=0.12, color=COLORS[0])
    ax5.set_xlabel("Weekly Budget (₹ Millions)"); ax5.set_ylabel("Max Profit (₹ Millions)")
    ax5.set_title("5. Sensitivity Analysis — Optimal Profit vs Weekly Budget", fontweight="bold")
    ax5.legend(); ax5.set_facecolor("#F8FAFC")
    ax5.grid(True, alpha=0.3)

    # ── 6. Supply-shock comparison ──
    ax6 = fig.add_subplot(gs[3, 0])
    base_units  = [lp["units"][p]   for p in PRODUCTS]
    shock_units = [shock["units"][p] for p in PRODUCTS]
    x_pos2 = np.arange(len(PRODUCTS))
    ax6.bar(x_pos2 - w/2, base_units,  w, label="Baseline",      color=COLORS[0], alpha=0.85)
    ax6.bar(x_pos2 + w/2, shock_units, w, label="Chip Shortage (−20%)", color=COLORS[3], alpha=0.85)
    ax6.set_xticks(x_pos2); ax6.set_xticklabels(PRODUCTS, fontsize=9)
    ax6.set_ylabel("Units/week")
    ax6.set_title("6. What-If: Semiconductor Supply\nShock (−20%)", fontweight="bold")
    ax6.legend(fontsize=8); ax6.set_facecolor("#F8FAFC")

    # ── 7. KPI summary card ──
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis("off")
    profit_drop = lp["total_profit"] - shock["total_profit"]
    kpis = [
        ["Metric", "Baseline", "After Shock"],
        ["Weekly Profit (₹M)", f"{lp['total_profit']/1e6:.2f}", f"{shock['total_profit']/1e6:.2f}"],
        ["Profit Drop (₹M)", "—", f"▼ {profit_drop/1e6:.2f}"],
        ["ILP Profit (₹M)", f"{ilp['total_profit']/1e6:.2f}", "—"],
        ["LP–ILP Gap (₹)", f"{(lp['total_profit']-ilp['total_profit']):.0f}", "—"],
    ]
    tbl = ax7.table(cellText=kpis[1:], colLabels=kpis[0],
                    cellLoc="center", loc="center",
                    colWidths=[0.45, 0.28, 0.27])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1.1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1E293B"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EFF6FF")
    ax7.set_title("7. KPI Summary", fontweight="bold", pad=16)
    ax7.set_facecolor("#F8FAFC")

    plt.suptitle("Production Planning Optimization — Kartikay Verma",
                 fontsize=16, fontweight="bold", y=0.99, color="#1E293B")

    path = os.path.join(OUT, "optimization_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close()
    print(f"Dashboard saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PRODUCTION PLANNING OPTIMIZATION — Kartikay Verma")
    print("=" * 60)

    print("\n[1/4] Solving LP (continuous) …")
    lp, _ = solve_lp()
    print(f"  Status  : {lp['status']}")
    print(f"  Profit  : ₹{lp['total_profit']:,.0f}")
    print(f"  Units   : {lp['units']}")

    print("\n[2/4] Solving ILP (integer) …")
    ilp = solve_ilp()
    print(f"  Status  : {ilp['status']}")
    print(f"  Profit  : ₹{ilp['total_profit']:,.0f}")
    print(f"  Units   : {ilp['units']}")

    print("\n[3/4] Running sensitivity analysis …")
    budgets, profits = sensitivity_budget()
    print(f"  Budget range: ₹{budgets[0]/1e6:.1f}M – ₹{budgets[-1]/1e6:.1f}M")

    print("\n[4/4] Supply shock scenario …")
    shock = solve_supply_shock()
    print(f"  Profit after shock: ₹{shock['total_profit']:,.0f}")
    print(f"  Profit drop       : ₹{lp['total_profit'] - shock['total_profit']:,.0f}")

    print("\nGenerating dashboard …")
    plot_all(lp, ilp, shock, budgets, profits)

    # Save summary JSON for notebook embedding
    summary = {"lp": lp, "ilp": ilp, "shock": shock,
                "sensitivity": {"budgets": budgets, "profits": profits}}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Results saved → results.json")
    print("\nDone ✓")
