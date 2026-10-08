import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Set global styles
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = ":"

# Load CV data
df1 = pd.read_csv(ROOT / "model_comparison_var1.csv")
df2 = pd.read_csv(ROOT / "model_comparison_var2.csv")

# Load training data
t1 = pd.read_csv(ROOT / "BT2024259" / "BT2024259_train_var1.csv")
t2 = pd.read_csv(ROOT / "BT2024259" / "BT2024259_train_var2.csv")

X1, y1 = t1.drop(columns="y").values, t1["y"].values
X2, y2 = t2.drop(columns="y").values, t2["y"].values

# 5-fold CV out-of-fold predictions
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Var 1 Winning Pipeline: Lasso deg 5, alpha 0.01
pipe1 = Pipeline([
    ("poly", PolynomialFeatures(degree=5, include_bias=False)),
    ("scaler", StandardScaler()),
    ("reg", Lasso(alpha=0.01, max_iter=100000))
])
oof1 = np.zeros_like(y1)
for tr, val in cv.split(X1, y1):
    pipe1.fit(X1[tr], y1[tr])
    oof1[val] = pipe1.predict(X1[val])

# Var 2 Winning Pipeline: Ridge deg 12, alpha 1.0
pipe2 = Pipeline([
    ("poly", PolynomialFeatures(degree=12, include_bias=False)),
    ("scaler", StandardScaler()),
    ("reg", Ridge(alpha=1.0))
])
oof2 = np.zeros_like(y2)
for tr, val in cv.split(X2, y2):
    pipe2.fit(X2[tr], y2[tr])
    oof2[val] = pipe2.predict(X2[val])

res1 = y1 - oof1
res2 = y2 - oof2

print("OOF 1 computed:", r2_score(y1, oof1), mean_squared_error(y1, oof1))
print("OOF 2 computed:", r2_score(y2, oof2), mean_squared_error(y2, oof2))

# ==============================================================================
# PLOT 1: CV MSE vs Degree (var1) & Residual Analysis (Side-by-side)
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1]})

# Subplot 1: CV MSE vs Degree for var1
methods_styles = [
    ("linear", None, "OLS Linear", "#d62728", "-", "o"),
    ("ridge", 0.1, r"Ridge ($\alpha=0.1$)", "#1f77b4", "--", "s"),
    ("ridge", 1.0, r"Ridge ($\alpha=1.0$)", "#3399e6", "--", "v"),
    ("ridge", 10.0, r"Ridge ($\alpha=10$)", "#004080", "--", "^"),
    ("lasso", 0.01, r"Lasso ($\alpha=0.01$)", "#ff7f0e", "-", "D"),
    ("lasso", 0.1, r"Lasso ($\alpha=0.1$)", "#ffa64d", ":", "d"),
    ("elastic_net", 0.01, r"EN ($\alpha=0.01$)", "#2ca02c", "-.", "p"),
]

for method, alpha, label, color, ls, marker in methods_styles:
    if alpha is None:
        sub = df1[df1["method"] == method]
    else:
        sub = df1[(df1["method"] == method) & (np.isclose(df1["alpha"], alpha))]
    sub = sub.sort_values("degree")
    ax1.plot(sub["degree"], sub["mse"], label=label, color=color, linestyle=ls, marker=marker, markersize=4, alpha=0.85)

ax1.set_yscale("log")
ax1.set_xlabel("Polynomial Degree", fontsize=10, fontweight="bold")
ax1.set_ylabel("CV MSE (log scale)", fontsize=10, fontweight="bold")
ax1.set_title("Part 1 (var1) - CV MSE vs Degree: All Models", fontsize=10.5, fontweight="bold")
ax1.grid(True)
ax1.set_xticks(range(1, 11))
ax1.plot(5, 0.3013, marker="*", markersize=14, color="#e6a100", markeredgecolor="black", zorder=10)
ax1.annotate("Selected model\nLasso (α=0.01, deg=5)\nCV MSE=0.3013",
             xy=(5, 0.3013), xytext=(5.3, 0.6),
             arrowprops=dict(facecolor="black", arrowstyle="->", lw=1),
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#cc9900", lw=0.8))
ax1.legend(fontsize=7, loc="upper right", framealpha=0.9)

# Subplot 2: Residuals vs Fitted
ax2.scatter(oof1, res1, alpha=0.45, color="#1f77b4", edgecolors="none", s=18)
ax2.axhline(0, color="red", linestyle="--", lw=1)
ax2.set_xlabel("Fitted Values", fontsize=10, fontweight="bold")
ax2.set_ylabel("Residuals", fontsize=10, fontweight="bold")
ax2.set_title("Residuals vs Fitted Values", fontsize=10.5, fontweight="bold")
ax2.grid(True)

# Subplot 3: Residual Distribution
count, bins, _ = ax2_dist = ax3.hist(res1, bins=35, density=True, alpha=0.5, color="#ff7f0e", edgecolor="white")
mu, std = norm.fit(res1)
xmin, xmax = ax3.get_xlim()
x_axis = np.linspace(xmin, xmax, 100)
p = norm.pdf(x_axis, mu, std)
ax3.plot(x_axis, p, "r-", lw=1.5, label=f"Normal (μ={mu:.3f}, σ={std:.3f})")
ax3.set_xlabel("Residual Value", fontsize=10, fontweight="bold")
ax3.set_ylabel("Density", fontsize=10, fontweight="bold")
ax3.set_title("Residual Distribution", fontsize=10.5, fontweight="bold")
ax3.grid(True)
ax3.legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig1_var1_mse_and_residuals.png", dpi=300)
plt.close(fig)
print("Saved fig1")

# ==============================================================================
# PLOT 2: CV MSE vs Degree for Problem 1 & Problem 2 side-by-side
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

for method, alpha, label, color, ls, marker in methods_styles:
    if alpha is None:
        sub = df1[df1["method"] == method]
    else:
        sub = df1[(df1["method"] == method) & (np.isclose(df1["alpha"], alpha))]
    sub = sub.sort_values("degree")
    ax1.plot(sub["degree"], sub["mse"], label=label, color=color, linestyle=ls, marker=marker, markersize=4, alpha=0.85)

ax1.set_yscale("log")
ax1.set_xlabel("Polynomial Degree", fontsize=10, fontweight="bold")
ax1.set_ylabel("CV MSE (log scale)", fontsize=10, fontweight="bold")
ax1.set_title("(a) Problem 1 (var1) - CV MSE vs Degree: All Models", fontsize=11, fontweight="bold")
ax1.grid(True)
ax1.set_xticks(range(1, 11))
ax1.plot(5, 0.3013, marker="*", markersize=14, color="#e6a100", markeredgecolor="black", zorder=10)
ax1.annotate("Selected: Lasso (α=0.01, deg=5)\nCV MSE=0.3013",
             xy=(5, 0.3013), xytext=(4.8, 0.65),
             arrowprops=dict(facecolor="black", arrowstyle="->", lw=1),
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#cc9900", lw=0.8))
ax1.legend(fontsize=7.5, loc="upper right")

# Subplot 2: Problem 2 (Regularised Models, degrees 1-20)
styles_var2 = [
    ("ridge", 0.01, r"Ridge ($\alpha=0.01$)", "#1f77b4", "--", "s"),
    ("ridge", 0.1, r"Ridge ($\alpha=0.1$)", "#3399e6", "--", "v"),
    ("ridge", 1.0, r"Ridge ($\alpha=1.0$)", "#004080", "-", "^"),
    ("ridge", 10.0, r"Ridge ($\alpha=10$)", "#001a33", "--", "<"),
    ("lasso", 0.01, r"Lasso ($\alpha=0.01$)", "#ff7f0e", "-", "D"),
    ("lasso", 0.1, r"Lasso ($\alpha=0.1$)", "#ffa64d", ":", "d"),
    ("elastic_net", 0.01, r"EN ($\alpha=0.01$)", "#2ca02c", "-.", "p"),
]

for method, alpha, label, color, ls, marker in styles_var2:
    sub = df2[(df2["method"] == method) & (np.isclose(df2["alpha"], alpha))].sort_values("degree")
    ax2.plot(sub["degree"], sub["mse"], label=label, color=color, linestyle=ls, marker=marker, markersize=4, alpha=0.85)

ax2.set_yscale("log")
ax2.set_xlabel("Polynomial Degree", fontsize=10, fontweight="bold")
ax2.set_ylabel("CV MSE (log scale)", fontsize=10, fontweight="bold")
ax2.set_title("(b) Problem 2 (var2) - CV MSE: Regularised Models (degrees 1-20)", fontsize=11, fontweight="bold")
ax2.grid(True)
ax2.set_xticks(range(1, 21, 2))
ax2.plot(12, 0.2566, marker="*", markersize=14, color="#e6a100", markeredgecolor="black", zorder=10)
ax2.annotate("Selected: Ridge (α=1.0, deg=12)\nCV MSE=0.2566",
             xy=(12, 0.2566), xytext=(10.5, 0.45),
             arrowprops=dict(facecolor="black", arrowstyle="->", lw=1),
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#cc9900", lw=0.8))
ax2.legend(fontsize=7.5, loc="upper right")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig2_cv_mse_comparison.png", dpi=300)
plt.close(fig)
print("Saved fig2")

# ==============================================================================
# PLOT 3: Best CV MSE per Degree Bar Chart (var1)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4.2))

best_per_deg1 = df1.loc[df1.groupby("degree")["mse"].idxmin()].sort_values("degree")
family_colors = {
    "linear": "#d62728",
    "ridge": "#1f77b4",
    "lasso": "#ff7f0e",
    "elastic_net": "#2ca02c"
}

bar_colors = [family_colors[m] for m in best_per_deg1["method"]]
bars = ax.bar(best_per_deg1["degree"], best_per_deg1["mse"], color=bar_colors, edgecolor="black", lw=0.6, width=0.65)

ax.set_xlabel("Polynomial Degree", fontsize=10.5, fontweight="bold")
ax.set_ylabel("Best CV MSE", fontsize=10.5, fontweight="bold")
ax.set_title("Part 1 (var1) - Best CV MSE per Degree (bar colour = winning model family)", fontsize=11, fontweight="bold")
ax.set_xticks(range(1, 11))
ax.grid(True, axis="y")

# Value labels on top of bars
for bar, m, mse_val in zip(bars, best_per_deg1["method"], best_per_deg1["mse"]):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + (0.15 if yval < 5 else 0.3),
            f"{mse_val:.3f}\n({m[:5]})", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# Custom legend for model family
handles = [plt.Rectangle((0, 0), 1, 1, color=family_colors[k], ec="black", lw=0.5) for k in ["linear", "ridge", "lasso", "elastic_net"]]
labels = ["OLS Linear", "Ridge", "Lasso", "ElasticNet"]
ax.legend(handles, labels, fontsize=8, loc="upper right")
ax.set_ylim(0, max(best_per_deg1["mse"]) * 1.15)
ax.axvline(5, color="red", linestyle="--", lw=1.2, alpha=0.7)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig3_var1_best_mse_bars.png", dpi=300)
plt.close(fig)
print("Saved fig3")

# ==============================================================================
# PLOT 4: Best CV MSE per Degree (var2)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4.2))

best_per_deg2 = df2.loc[df2.groupby("degree")["mse"].idxmin()].sort_values("degree")
bar_colors2 = [family_colors[m] for m in best_per_deg2["method"]]
bars = ax.bar(
    best_per_deg2["degree"],
    best_per_deg2["mse"],
    color=bar_colors2,
    edgecolor="black",
    lw=0.6,
    width=0.65,
)

ax.set_xlabel("Polynomial Degree", fontsize=10.5, fontweight="bold")
ax.set_ylabel("Best CV MSE", fontsize=10.5, fontweight="bold")
ax.set_title("Problem 2 (var2) - Best CV MSE per Degree", fontsize=11, fontweight="bold")
ax.set_xticks(range(1, 21))
ax.grid(True, axis="y")

for bar, method, mse_value in zip(
    bars, best_per_deg2["method"], best_per_deg2["mse"]
):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.08,
        f"{mse_value:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        rotation=90,
    )

handles = [
    plt.Rectangle((0, 0), 1, 1, color=family_colors[k], ec="black", lw=0.5)
    for k in ["linear", "ridge", "lasso", "elastic_net"]
]
ax.legend(handles, ["OLS Linear", "Ridge", "Lasso", "ElasticNet"], fontsize=8, loc="upper right")
ax.axvline(12, color="red", linestyle="--", lw=1.2, alpha=0.7)
ax.annotate(
    "Best: Ridge, degree 12",
    xy=(12, best_per_deg2.loc[best_per_deg2["degree"] == 12, "mse"].iloc[0]),
    xytext=(13, 1.5),
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1),
    fontsize=8,
)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig6_var2_best_mse_bars.png", dpi=300)
plt.close(fig)
print("Saved fig6")

# ==============================================================================
# PLOT 5: Actual vs Predicted Scatter Plots (Problem 1 & Problem 2)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Var 1 Actual vs Predicted
abs_res1 = np.abs(res1)
sc1 = ax1.scatter(y1, oof1, c=abs_res1, cmap="coolwarm", s=22, alpha=0.8, edgecolors="none")
min_val1, max_val1 = min(y1.min(), oof1.min()) - 1, max(y1.max(), oof1.max()) + 1
ax1.plot([min_val1, max_val1], [min_val1, max_val1], "k--", lw=1.2, label="Perfect fit (y = ŷ)")
cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("|Residual|", fontsize=9)
ax1.set_xlabel("Actual y", fontsize=10, fontweight="bold")
ax1.set_ylabel("Predicted y", fontsize=10, fontweight="bold")
ax1.set_title("(a) Problem 1 (var1) - Actual vs Predicted\n(Lasso α=0.01, Degree 5, 5-Fold CV)", fontsize=10.5, fontweight="bold")
ax1.text(0.05, 0.92, f"CV R² = {r2_score(y1, oof1):.4f}\nCV MSE = {mean_squared_error(y1, oof1):.4f}",
         transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
ax1.set_xlim(min_val1, max_val1)
ax1.set_ylim(min_val1, max_val1)
ax1.grid(True)
ax1.legend(loc="lower right", fontsize=8.5)

# Var 2 Actual vs Predicted
abs_res2 = np.abs(res2)
sc2 = ax2.scatter(y2, oof2, c=abs_res2, cmap="coolwarm", s=22, alpha=0.8, edgecolors="none")
min_val2, max_val2 = min(y2.min(), oof2.min()) - 2, max(y2.max(), oof2.max()) + 2
ax2.plot([min_val2, max_val2], [min_val2, max_val2], "k--", lw=1.2, label="Perfect fit (y = ŷ)")
cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("|Residual|", fontsize=9)
ax2.set_xlabel("Actual y", fontsize=10, fontweight="bold")
ax2.set_ylabel("Predicted y", fontsize=10, fontweight="bold")
ax2.set_title("(b) Problem 2 (var2) - Actual vs Predicted\n(Ridge α=1.0, Degree 12, 5-Fold CV)", fontsize=10.5, fontweight="bold")
ax2.text(0.05, 0.92, f"CV R² = {r2_score(y2, oof2):.4f}\nCV MSE = {mean_squared_error(y2, oof2):.4f}",
         transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
ax2.set_xlim(min_val2, max_val2)
ax2.set_ylim(min_val2, max_val2)
ax2.grid(True)
ax2.legend(loc="lower right", fontsize=8.5)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig4_actual_vs_predicted.png", dpi=300)
plt.close(fig)
print("Saved fig4")

# ==============================================================================
# PLOT 6: CV R2 vs Degree for Problem 1 & Problem 2
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

for method, alpha, label, color, ls, marker in methods_styles:
    if alpha is None:
        sub = df1[df1["method"] == method]
    else:
        sub = df1[(df1["method"] == method) & (np.isclose(df1["alpha"], alpha))]
    sub = sub.sort_values("degree")
    ax1.plot(sub["degree"], sub["r2"], label=label, color=color, linestyle=ls, marker=marker, markersize=4, alpha=0.85)

ax1.set_xlabel("Polynomial Degree", fontsize=10, fontweight="bold")
ax1.set_ylabel("CV R²", fontsize=10, fontweight="bold")
ax1.set_title("(a) Problem 1 (var1) - CV R² vs Degree: All Models", fontsize=11, fontweight="bold")
ax1.set_ylim(-0.1, 1.02)
ax1.grid(True)
ax1.set_xticks(range(1, 11))
ax1.plot(5, 0.9703, marker="*", markersize=14, color="#e6a100", markeredgecolor="black", zorder=10)
ax1.legend(fontsize=7.5, loc="lower right")

for method, alpha, label, color, ls, marker in styles_var2:
    sub = df2[(df2["method"] == method) & (np.isclose(df2["alpha"], alpha))].sort_values("degree")
    ax2.plot(sub["degree"], sub["r2"], label=label, color=color, linestyle=ls, marker=marker, markersize=4, alpha=0.85)

ax2.set_xlabel("Polynomial Degree", fontsize=10, fontweight="bold")
ax2.set_ylabel("CV R²", fontsize=10, fontweight="bold")
ax2.set_title("(b) Problem 2 (var2) - CV R²: Regularised Models (degrees 1-20)", fontsize=11, fontweight="bold")
ax2.set_ylim(0.88, 1.002)
ax2.grid(True)
ax2.set_xticks(range(1, 21, 2))
ax2.plot(12, 0.9935, marker="*", markersize=14, color="#e6a100", markeredgecolor="black", zorder=10)
ax2.legend(fontsize=7.5, loc="lower right")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig5_cv_r2_curves.png", dpi=300)
plt.close(fig)
print("Saved fig5")

# ==============================================================================
# PLOT 7: Residual Diagnostics for Problem 2
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

ax1.scatter(oof2, res2, alpha=0.45, color="#1f77b4", edgecolors="none", s=18)
ax1.axhline(0, color="red", linestyle="--", lw=1)
ax1.set_xlabel("Fitted Values", fontsize=10, fontweight="bold")
ax1.set_ylabel("Residuals", fontsize=10, fontweight="bold")
ax1.set_title("Problem 2: Residuals vs Fitted Values", fontsize=10.5, fontweight="bold")
ax1.grid(True)

ax2.hist(res2, bins=35, density=True, alpha=0.5, color="#ff7f0e", edgecolor="white")
mu2, std2 = norm.fit(res2)
xmin2, xmax2 = ax2.get_xlim()
x_axis2 = np.linspace(xmin2, xmax2, 100)
ax2.plot(x_axis2, norm.pdf(x_axis2, mu2, std2), "r-", lw=1.5,
         label=f"Normal (mean={mu2:.3f}, std={std2:.3f})")
ax2.set_xlabel("Residual Value", fontsize=10, fontweight="bold")
ax2.set_ylabel("Density", fontsize=10, fontweight="bold")
ax2.set_title("Problem 2: Residual Distribution", fontsize=10.5, fontweight="bold")
ax2.grid(True)
ax2.legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "fig_residuals_var2.png", dpi=300)
plt.close(fig)
print("Saved fig_residuals_var2")

print("All figures generated successfully!")
