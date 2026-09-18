"""Deterministic, CPU-only optimization experiments; no downloads or notebooks needed."""
import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np


def solve(matrix, initial, method="fixed", step=None, max_steps=2000, tolerance=1e-8):
    """Minimize x.T A x / 2. Return the initial point and every accepted iterate.

    Heavy-ball uses the optimal *quadratic-specific* spectral parameters, not a
    universally tuned optimizer. Armijo counts rejected objective evaluations.
    """
    a = np.asarray(matrix, dtype=float)
    x = np.asarray(initial, dtype=float).copy()
    if a.ndim != 2 or a.shape != (x.size, x.size) or x.ndim != 1:
        raise ValueError("matrix and initial must have matching dimensions")
    if not np.isfinite(a).all() or not np.isfinite(x).all() or not np.allclose(a, a.T):
        raise ValueError("finite symmetric matrix and finite initial point required")
    eigenvalues = np.linalg.eigvalsh(a)
    mu, lipschitz = eigenvalues[0], eigenvalues[-1]
    if mu <= 0 or method not in {"fixed", "momentum", "armijo"}:
        raise ValueError("positive-definite matrix and known method required")
    if max_steps < 0 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("invalid iteration budget or tolerance")
    rate = 1 / lipschitz if step is None else float(step)
    if rate <= 0 or not np.isfinite(rate):
        raise ValueError("step must be finite and positive")
    beta = 0.0
    if method == "momentum":
        root_l, root_m = np.sqrt(lipschitz), np.sqrt(mu)
        rate = 4 / (root_l + root_m) ** 2
        beta = ((root_l - root_m) / (root_l + root_m)) ** 2
    previous = x.copy()
    objective = lambda point: float(point @ a @ point / 2)
    rows = []
    evaluations = 0
    status = "budget_exhausted"
    for iteration in range(max_steps + 1):
        value = objective(x)
        evaluations += 1
        gradient = a @ x
        norm = float(np.linalg.norm(gradient))
        rows.append({"iteration": iteration, "objective": value, "gradient_norm": norm,
                     "objective_evaluations": evaluations, "point": x.tolist()})
        if norm <= tolerance:
            status = "converged"
            break
        if value > 1e30 or not np.isfinite(value):
            status = "diverged"
            break
        if iteration == max_steps:
            break
        alpha = rate
        if method == "armijo":
            for _ in range(80):
                trial = x - alpha * gradient
                trial_value = objective(trial)
                evaluations += 1
                if trial_value <= value - 1e-4 * alpha * norm**2:
                    break
                alpha *= 0.5
            else:
                status = "line_search_failed"
                break
        candidate = x - alpha * gradient + beta * (x - previous)
        previous, x = x, candidate
    return {"status": status, "condition_number": float(lipschitz / mu),
            "step": rate, "momentum": float(beta), "trace": rows}


def run_suite(seed=7):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    initial = np.array([4.0, -3.0])
    results = {}
    for condition in (1, 10, 100):
        matrix = rotation @ np.diag([1.0, float(condition)]) @ rotation.T
        for method, rate in (("fixed", 1 / condition), ("momentum", None), ("armijo", 1.0)):
            results[f"kappa_{condition}/{method}"] = solve(matrix, initial, method, rate)
        results[f"kappa_{condition}/unstable"] = solve(matrix, initial, step=2.1 / condition)
    return results


def export(results, output, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps({
        "seed": seed, "python": platform.python_version(), "numpy": np.__version__,
        "objective": "0.5 * x.T @ A @ x; exact optimum x=0, f=0",
        "stopping_rule": "gradient L2 norm <= 1e-8; at most 2000 updates",
        "results": results}, indent=2, allow_nan=False) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "status", "updates", "objective", "gradient_norm", "objective_evaluations"])
        for name, run in results.items():
            last = run["trace"][-1]
            writer.writerow([name, run["status"], last["iteration"], last["objective"],
                             last["gradient_norm"], last["objective_evaluations"]])
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    colors = {"fixed": "#4068ac", "momentum": "#14857c", "armijo": "#aa6c25", "unstable": "#b14e52"}
    for ax, condition in zip(axes, (1, 10, 100)):
        for name, run in results.items():
            if name.split("/")[0] != f"kappa_{condition}":
                continue
            method = name.split("/")[1]
            trace = run["trace"]
            ax.semilogy([r["iteration"] for r in trace],
                        [max(r["objective"], 1e-22) for r in trace],
                        label=method, color=colors[method], linewidth=1.7)
        ax.set(title=f"Condition number {condition}", xlabel="Updates", ylabel="Objective gap")
        ax.set_ylim(1e-22, 1e6)
        ax.grid(alpha=0.16)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output / "convergence.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    suite = run_suite(args.seed)
    export(suite, args.output, args.seed)
    for name, result in suite.items():
        print(f"{name:22s} {result['status']:18s} updates={result['trace'][-1]['iteration']}")
