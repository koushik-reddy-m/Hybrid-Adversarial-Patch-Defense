
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from src.config import RESULTS_DIR, PLOTS_DIR, GRADCAM_DIR


METHOD_LABELS = {
    "no_defense":      "No defense",
    "jpeg":            "JPEG compression",
    "feature_squeeze": "Feature squeezing",
    "gradcam_only":    "GradCAM only",
    "hybrid":          "Hybrid pipeline",
}

COLORS = {
    "clean_acc":     "#378ADD",
    "recovered_acc": "#1D9E75",
    "fpr":           "#D85A30",
}

import numpy as np
from scipy import stats

def bootstrap_confidence_interval(
    successes: int,
    total: int,
    n_bootstrap: int = 1000,
    ci: float = 95
) -> tuple[float, float, float]:

    if total == 0:
        return 0.0, 0.0, 0.0
    
    point_estimate = 100 * successes / total
    
    bootstrap_rates = []
    np.random.seed(42)  
    
    for _ in range(n_bootstrap):
        bootstrap_successes = np.random.choice(
            [0, 1], 
            size=total, 
            p=[1 - successes/total, successes/total]
        ).sum()
        bootstrap_rates.append(100 * bootstrap_successes / total)
    
    alpha = 100 - ci
    lower = np.percentile(bootstrap_rates, alpha/2)
    upper = np.percentile(bootstrap_rates, 100 - alpha/2)
    
    return point_estimate, lower, upper


def bootstrap_ci_metric_pair(
    clean_successes: int,
    patched_successes: int,
    fp_successes: int,
    total: int,
    n_bootstrap: int = 1000
) -> dict:
    
    clean_est, clean_low, clean_high = bootstrap_confidence_interval(
        clean_successes, total, n_bootstrap
    )
    
    recovered_est, recovered_low, recovered_high = bootstrap_confidence_interval(
        patched_successes, total, n_bootstrap
    )
    
    fpr_est, fpr_low, fpr_high = bootstrap_confidence_interval(
        fp_successes, total, n_bootstrap
    )
    
    attack_est = 100 - recovered_est
    attack_low = 100 - recovered_high
    attack_high = 100 - recovered_low
    
    return {
        "clean_acc": {"value": clean_est, "ci_low": clean_low, "ci_high": clean_high},
        "attack_sr": {"value": attack_est, "ci_low": attack_low, "ci_high": attack_high},
        "recovered_acc": {"value": recovered_est, "ci_low": recovered_low, "ci_high": recovered_high},
        "fpr": {"value": fpr_est, "ci_low": fpr_low, "ci_high": fpr_high},
    }


def print_results_with_ci(results: dict):
    print("\n" + "=" * 85)
    print(f"{'Method':<22} {'Clean acc':>15} {'Attack SR':>15} {'Recovered':>15} {'FPR':>12}")
    print("-" * 85)
    
    order = ["no_defense", "jpeg", "feature_squeeze", "gradcam_only", "hybrid"]
    
    for method in order:
        if method not in results:
            continue
        
        r = results[method]
        marker = " ★" if method == "hybrid" else ""
        
        clean_str = f"{r['clean_acc']:.1f}% [{r['clean_ci_low']:.1f}-{r['clean_ci_high']:.1f}]"
        attack_str = f"{r['attack_sr']:.1f}% [{r['attack_ci_low']:.1f}-{r['attack_ci_high']:.1f}]"
        rec_str = f"{r['recovered_acc']:.1f}% [{r['rec_ci_low']:.1f}-{r['rec_ci_high']:.1f}]"
        fpr_str = f"{r['fpr']:.1f}% [{r['fpr_ci_low']:.1f}-{r['fpr_ci_high']:.1f}]"
        
        print(f"{method + marker:<22} {clean_str:>15} {attack_str:>15} {rec_str:>15} {fpr_str:>12}")
    
    print("=" * 85)
    print("Note: 95% bootstrap confidence intervals in brackets\n")


def load_results() -> dict:
    """Load saved results from disk."""
    path = os.path.join(RESULTS_DIR, "results.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results at {path}. Run run_full_evaluation() first.")
    results = torch.load(path, map_location="cpu")
    print(f"Results loaded from {path}")
    return results


def generate_all_plots(results: dict | None = None):
    """Generate all output figures from a results dict (or load from disk)."""
    if results is None:
        results = load_results()

    _plot_results_table_with_ci(results)
    _plot_grouped_bars(results)
    _plot_radar(results)
    _plot_gradcam_grid()
    print(f" All plots saved to {PLOTS_DIR}/")


#  Results table bar chart 

def _plot_results_table_with_ci(results: dict):
    methods = [m for m in ["no_defense", "jpeg", "feature_squeeze", 
                            "gradcam_only", "hybrid"] if m in results]
    labels = [METHOD_LABELS[m] for m in methods]
    n = len(methods)

    clean_vals = [results[m]["clean_acc"] for m in methods]
    clean_errs = [[results[m]["clean_acc"] - results[m]["clean_ci_low"],
                   results[m]["clean_ci_high"] - results[m]["clean_acc"]] for m in methods]
    
    recovered_vals = [results[m]["recovered_acc"] for m in methods]
    recovered_errs = [[results[m]["recovered_acc"] - results[m]["rec_ci_low"],
                       results[m]["rec_ci_high"] - results[m]["recovered_acc"]] for m in methods]
    
    fpr_vals = [results[m]["fpr"] for m in methods]
    fpr_errs = [[results[m]["fpr"] - results[m]["fpr_ci_low"],
                 results[m]["fpr_ci_high"] - results[m]["fpr"]] for m in methods]

    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    
    b1 = ax.bar(x - w, clean_vals, w, label="Clean accuracy %", 
                color=COLORS["clean_acc"], yerr=[e[0] for e in clean_errs], 
                capsize=3, error_kw={'linewidth': 1, 'color': 'black'})
    b2 = ax.bar(x, recovered_vals, w, label="Recovered accuracy %", 
                color=COLORS["recovered_acc"], yerr=[e[0] for e in recovered_errs],
                capsize=3, error_kw={'linewidth': 1, 'color': 'black'})
    b3 = ax.bar(x + w, fpr_vals, w, label="False positive rate %", 
                color=COLORS["fpr"], yerr=[e[0] for e in fpr_errs],
                capsize=3, error_kw={'linewidth': 1, 'color': 'black'})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Defense comparison with 95% confidence intervals")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "results_table_with_ci.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Results with CI plot {path}")

def _plot_grouped_bars(results: dict):
    methods = [m for m in ["no_defense", "jpeg", "feature_squeeze",
                            "gradcam_only", "hybrid"] if m in results]
    labels  = [METHOD_LABELS[m] for m in methods]

    recovered = [results[m]["recovered_acc"] for m in methods]
    clean     = [results[m]["clean_acc"]     for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    colours_r = ["#D85A30" if m != "hybrid" else "#1D9E75" for m in methods]
    axes[0].barh(labels, recovered, color=colours_r, zorder=3)
    axes[0].set_xlabel("Recovered accuracy (%)")
    axes[0].set_title("Accuracy on patched images after defense")
    axes[0].set_xlim(0, 100)
    axes[0].grid(axis="x", alpha=0.3, zorder=0)
    for i, v in enumerate(recovered):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    colours_c = ["#378ADD" if m != "hybrid" else "#534AB7" for m in methods]
    axes[1].barh(labels, clean, color=colours_c, zorder=3)
    axes[1].set_xlabel("Clean accuracy (%)")
    axes[1].set_title("Accuracy on clean images (no patch)")
    axes[1].set_xlim(0, 100)
    axes[1].grid(axis="x", alpha=0.3, zorder=0)
    for i, v in enumerate(clean):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "attack_vs_defense.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Attack vs defense plot {path}")


def _plot_radar(results: dict):
    methods = [m for m in ["no_defense", "jpeg", "feature_squeeze",
                            "gradcam_only", "hybrid"] if m in results]

    categories  = ["Clean acc", "Recovered acc", "Low FPR"]
    N_cats      = len(categories)
    angles      = [n / N_cats * 2 * np.pi for n in range(N_cats)]
    angles     += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    palette = ["#888780", "#378ADD", "#534AB7", "#EF9F27", "#1D9E75"]

    for idx, method in enumerate(methods):
        r = results[method]
        values = [r["clean_acc"], r["recovered_acc"], 100 - r["fpr"]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5,
                color=palette[idx], label=METHOD_LABELS[method])
        ax.fill(angles, values, alpha=0.07, color=palette[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8)
    ax.set_title("Defense comparison:radar view", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "radar_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Radar chart {path}")



def _plot_gradcam_grid():
    example_paths = sorted(glob.glob(os.path.join(GRADCAM_DIR, "example_*.png")))
    if not example_paths:
        print("[evaluate] No GradCAM examples found,skipping grid.")
        return

    n      = len(example_paths)
    ncols  = 2
    nrows  = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    axes = np.array(axes).flatten()

    for i, path in enumerate(example_paths):
        img = Image.open(path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Example {i+1}", fontsize=10)

    for j in range(len(example_paths), len(axes)):
        axes[j].axis("off")

    plt.suptitle("GradCAM localisation + masking examples", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "gradcam_grid.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] GradCAM grid  {path}")


def print_results(results: dict | None = None):
    if results is None:
        results = load_results()
    methods = [m for m in ["no_defense", "jpeg", "feature_squeeze",
                            "gradcam_only", "hybrid"] if m in results]
    print("\n" + "-" * 75)
    print(f"{'Method':<26} {'Clean':>7} {'Atk SR':>8} {'Recov':>8} {'FPR':>7}")
    print("-" * 75)
    for m in methods:
        r = results[m]
        mark = " *" if m == "hybrid" else ""
        print(f"{METHOD_LABELS[m] + mark:<26} {r['clean_acc']:>6.1f}% "
              f"{r['attack_sr']:>7.1f}% "
              f"{r['recovered_acc']:>7.1f}% "
              f"{r['fpr']:>6.1f}%")
    print("-" * 75 + "\n")
