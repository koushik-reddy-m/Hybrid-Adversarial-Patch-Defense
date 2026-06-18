import numpy as np
from typing import Tuple, Dict, Any


def bootstrap_ci(
    successes: int,
    total: int,
    n_bootstrap: int = 10000,
    ci: float = 95,
    random_state: int = 42
) -> Tuple[float, float]:

    if total == 0:
        return (0.0, 0.0)
    
    np.random.seed(random_state)
    proportion = successes / total
    
    # Bootstrap resampling
    bootstrap_proportions = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(total, size=total, replace=True)
        bootstrap_successes = np.sum(indices < successes)
        bootstrap_proportions.append(bootstrap_successes / total)
    
    # Percentile-based CI
    alpha = 100 - ci
    lower = np.percentile(bootstrap_proportions, alpha / 2)
    upper = np.percentile(bootstrap_proportions, 100 - alpha / 2)
    
    return (lower * 100, upper * 100)


def proportion_ci_wilson(
    successes: int,
    total: int,
    ci: float = 95
) -> Tuple[float, float]:

    if total == 0:
        return (0.0, 0.0)
    
    from scipy import stats
    
    p = successes / total
    z = stats.norm.ppf(1 - (100 - ci) / 200)  # e.g., 1.96 for 95% CI
    
    denominator = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    
    lower = (centre - half_width) / denominator
    upper = (centre + half_width) / denominator
    
    return (lower * 100, upper * 100)


def compare_proportions(
    successes1: int,
    total1: int,
    successes2: int,
    total2: int,
) -> Dict[str, Any]:
    from scipy.stats import chi2_contingency
    
    table = np.array([
        [successes1, total1 - successes1],
        [successes2, total2 - successes2]
    ])
    
    chi2, p_value, dof, expected = chi2_contingency(table)
    
    n = total1 + total2
    phi = np.sqrt(chi2 / n)
    
    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "effect_size": phi,
        "chi2": chi2
    }


def mean_ci(
    values: list,
    ci: float = 95,
    method: str = "bootstrap"
) -> Tuple[float, float, float]:

    mean = np.mean(values)
    n = len(values)
    
    if method == "t" and n > 1:
        from scipy import stats
        sem = stats.sem(values)
        h = sem * stats.t.ppf(1 - (100 - ci) / 200, n - 1)
        return (mean, mean - h, mean + h)
    else:
        # Bootstrap
        np.random.seed(42)
        bootstrap_means = []
        for _ in range(10000):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 100 - ci
        lower = np.percentile(bootstrap_means, alpha / 2)
        upper = np.percentile(bootstrap_means, 100 - alpha / 2)
        return (mean, lower, upper)


def format_ci(value: float, lower: float, upper: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"