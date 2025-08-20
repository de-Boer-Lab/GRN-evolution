
# import evolutionMatrix
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import os, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]

sys.path.append(str(Path(__file__).parent.parent / "simulation-code" / "static-environment"))
import evoHelpers as ev
# Directories (relative to repo root)
figure_directory = repo_root / "analyses" / "figures"


STD_THRESHOLD = 0.025
N_SAMPLES = 10000

def compute_expressions(n_TFs, n_targets, edge_weight, n_samples=N_SAMPLES): 
    regulator_list = list(range(n_TFs))
    activator_list = list(range(n_TFs//2))
    repressor_list = list(range(n_TFs // 2, n_TFs))
    
    ind = np.full((n_TFs, n_targets + n_TFs), edge_weight)
    
    geneList = np.zeros(n_targets + n_TFs)
    end_expression_levels = []
    
    for _ in range(n_samples): 
        expression = [random.uniform(-1,1) for gene in geneList]
        
        stable = False
        initial_expression = ev.setExpression(regulator_list, activator_list, repressor_list, expression, ind)
        
        while not stable: 
            new_expression = ev.setExpression(regulator_list, activator_list, repressor_list, expression, ind)
            if np.allclose(initial_expression, new_expression, atol=1e-4):
                stable = True
            else:
                initial_expression = new_expression
                
        end_expression_levels.append(new_expression[0])
        
    return np.array(end_expression_levels)

def std_vs_edge_weight(n_TFs, n_targets, edge_weight_grid):
    stds = []
    dists = []
    for k in edge_weight_grid: 
        print("Edge weight: ", k)
        expressions = compute_expressions(n_TFs, n_targets, k)
        stds.append(np.std(expressions))
        dists.append(expressions)
    return np.array(stds), dists

def find_cutoff(stds, edge_weight_grid, threshold = 0.025):
    valid = np.where(stds <= threshold)[0]
    return edge_weight_grid[valid[-1]] if valid.size else np.nan
     

# Panel A:
number_of_regulators = 20 
number_of_targets = 10
edge_weights = np.arange(-55, -34, 1) / 10.0
print(edge_weights)
stds_A, dists_A = std_vs_edge_weight(number_of_regulators, number_of_targets, edge_weights)  
cutoff_A = find_cutoff(stds_A, edge_weights)

# Density plot? 
to_plot = []
if np.isfinite(cutoff_A):
    idx = np.where(edge_weights == cutoff_A)[0]
    if idx.size: 
        i = idx[0]
        
        picks = sorted(set([max(0, i-2), max(0, i-1), i, min(len(edge_weights)-1, i+1)]))
        to_plot = picks
if not to_plot:
    # fallback: just first four
    to_plot = list(range(0, min(4, len(edge_weights))))

fig = plt.figure(figsize=(10,6), constrained_layout = True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad = 0.02, wspace=0.08)
gs = gridspec.GridSpec(2,2, width_ratios=[1.6,1], height_ratios=[1,1], figure=fig)
ax_main = fig.add_subplot(gs[:,0])
ax_dist = fig.add_subplot(gs[0,1])
ax_cutoff = fig.add_subplot(gs[1,1])

# Top: densities
for i in to_plot:
    ln_k = edge_weights[i]
    sns.kdeplot(dists_A[i], ax=ax_dist, label=f"{ln_k:.2f}")
ax_dist.set_xlabel("Expression levels [ln(E)]")
ax_dist.set_ylabel("density")
ax_dist.legend(title='Binding affinity')

# Right: std vs ln(k)
ax_cutoff.scatter(edge_weights, stds_A, lw=1.5)
ax_cutoff.axhline(0.025, ls="--", lw=1, color="k")
if np.isfinite(cutoff_A):
    y = np.interp(cutoff_A, edge_weights, stds_A)
    ax_cutoff.scatter([cutoff_A], [y], s=30, zorder=3)
    ax_cutoff.annotate(f"cutoff\nln(k)={cutoff_A:.2f}", (cutoff_A, y), xytext=(6, -10),
                textcoords="offset points", fontsize=8)
ax_cutoff.set_xlabel("Binding affinity [ln(k)]")
ax_cutoff.set_ylabel("Expression σ")

ax_main.text(-0.10, 1.03, "A)", transform=ax_main.transAxes,
             fontsize=12, fontweight="bold", va="bottom", ha="right")
ax_dist.text(-0.10, 1.03, "B)", transform=ax_dist.transAxes,
             fontsize=12, fontweight="bold", va="bottom", ha="right")
ax_cutoff.text(-0.10, 1.03, "C)", transform=ax_cutoff.transAxes,
               fontsize=12, fontweight="bold", va="bottom", ha="right")


# Sweep several TFs and fit the line 

tf_counts = [10, 20, 30, 50, 75, 100, 150, 200]
cutoffs = []

for n in tf_counts:
    stds, _ = std_vs_edge_weight(n, number_of_targets, edge_weights)
    cutoff = find_cutoff(stds, edge_weights)
    cutoffs.append(cutoff)
    
    print(f"  n={n:3d} -> cutoff ln(k) = {cutoff}")
    
tf_counts = np.array(tf_counts, dtype=float)
cutoffs  = np.array(cutoffs, dtype=float)

# Fit ln(k) ~ a * ln(n) + b using only finite points
mask = np.isfinite(cutoffs)
ln_n = np.log(tf_counts[mask])
y = cutoffs[mask]
a_fit, b_fit = np.polyfit(ln_n, y, 1)

print("Equation: ", (3,5,"y = {0}*x+{1}".format(round(a_fit,3), round(b_fit,3))))


# R^2
y_hat = a_fit * ln_n + b_fit
r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2) if len(y) > 1 else np.nan

# Plot
ax_main.scatter(tf_counts, cutoffs, zorder=3)
# ax_main.plot(tf_counts, a_fit*np.log(tf_counts) + b_fit, lw=1.3,
#         label=f"fit: ln(k)={a_fit:.3f}·ln(n)+{b_fit:.3f} (R²={r2:.3f})")
ax_main.plot(tf_counts, a_fit*np.log(tf_counts) + b_fit, ls="--", lw=1, label="ln(k)={a_fit:.3f}·ln(n)+{b_fit:.3f}")

ax_main.set_xscale("log")
ax_main.set_xlabel("Number of TFs")
ax_main.set_ylabel("Cutoff ln(k)")
ax_main.legend(frameon=False)

# plt.savefig("panelB_cutoff_vs_n.png", dpi=300)
plt.savefig(os.path.join(figure_directory, 'Supp10.png'), dpi=1200)
plt.close()
