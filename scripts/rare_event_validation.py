"""
Rare-event validation: compare log P(Q_max >= B) estimates from
(a) direct Monte Carlo, (b) importance-sampled MC, and (c) the
Freidlin-Wentzell rate function I(phi*).

Usage:
    python scripts/rare_event_validation.py --n-paths 10000 --B-values 2 4 6 8

Outputs:
    results/rare_event/rate_function_validation.json
    results/rare_event/rate_function_validation.png
"""

import argparse
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Rate function computation
# ---------------------------------------------------------------------------

def freidlin_wentzell_optimal_path(lam, mu, B, T, n_points=200):
    """
    Numerically solve for the Freidlin-Wentzell optimal path phi* that
    minimises I(phi) = 0.5 * integral((dphi/dt - (lam-mu))^2 / lam) dt
    subject to max(phi) = B and phi(0) = 0, phi >= 0.

    For constant lambda(t) = lam, the optimal path is linear:
        phi*(t) = B * t / t*    for t <= t*
        phi*(t) = B             for t > t* (clamped at boundary)
    where t* is chosen so that I(phi*) is minimised.

    With constant drift d = lam - mu and constant sigma^2 = lam:
        I(phi_linear) = 0.5 * (v - d)^2 / lam * t*
    where v = B / t* is the path slope.

    Minimising over t* gives: t* = B / d when d > 0, else t* = B / sqrt(lam) * some scale.
    """
    d = lam - mu  # drift
    ts = np.linspace(0, T, n_points)
    dt = T / (n_points - 1)

    # Grid search over t_star (time to reach peak B)
    # For each candidate t_star, compute I(phi*)
    t_stars = np.linspace(0.1, T, 100)
    best_I = np.inf
    best_tstar = T

    for t_star in t_stars:
        v = B / t_star  # slope of ascending phase
        # Ascending phase [0, t_star]: dphi/dt = v
        I_asc = 0.5 * (v - d)**2 / lam * t_star
        # Descending or flat phase: for simplicity, assume phi stays at B after t_star
        # (drift pulls it down, so the IS path needs no additional energy)
        I_total = I_asc
        if I_total < best_I:
            best_I = I_total
            best_tstar = t_star

    # Reconstruct optimal path
    phi_star = np.zeros(n_points)
    for k, t in enumerate(ts):
        if t <= best_tstar:
            phi_star[k] = B * t / best_tstar
        else:
            phi_star[k] = B  # stays at peak (can descend, but this gives upper bound on I)

    return best_I, phi_star, ts


def rate_function_value(phi, ts, lam, mu):
    """Numerically evaluate I(phi) for a given path."""
    dt = ts[1] - ts[0]
    dphi = np.gradient(phi, ts)
    d = lam - mu
    integrand = (dphi - d)**2 / lam
    return 0.5 * np.trapz(integrand, ts)


# ---------------------------------------------------------------------------
# Monte Carlo estimators
# ---------------------------------------------------------------------------

def direct_mc_overflow_prob(lam, mu, B, T, n_paths, dt, seed):
    """Estimate P(Q_max >= B) using direct Monte Carlo (TV-sigma SDE)."""
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    sigma = np.sqrt(lam)
    d = lam - mu

    Q = np.zeros(n_paths)
    overflow = np.zeros(n_paths, dtype=bool)

    for _ in range(n_steps):
        noise = rng.normal(0.0, sqrt_dt, n_paths)
        Q = np.maximum(0.0, Q + d * dt + sigma * noise)
        overflow |= (Q >= B)

    prob = overflow.mean()
    se = np.sqrt(prob * (1 - prob) / n_paths) if prob > 0 else 0.0
    return float(prob), float(se)


def importance_sampled_mc(lam, mu, B, T, n_paths, dt, seed):
    """
    Estimate P(Q_max >= B) using importance sampling with the Girsanov
    change of measure toward the Freidlin-Wentzell optimal path.

    The IS drift h*(t) = (dphi*(t)/dt - (lam - mu)) / sqrt(lam).
    For the linear optimal path phi*(t) = B*t/t_star:
        h*(t) = (B/t_star - d) / sigma    for t <= t_star
        h*(t) = 0                           for t > t_star
    """
    rng = np.random.default_rng(seed + 50000)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    sigma = np.sqrt(lam)
    d = lam - mu

    # Get optimal path parameters
    best_I, phi_star, ts = freidlin_wentzell_optimal_path(lam, mu, B, T, n_steps + 1)

    # Compute h* at each step (derivative of phi_star)
    dphi = np.gradient(phi_star, ts)
    h_star = (dphi - d) / sigma  # IS perturbation in units of Brownian motion

    Q = np.zeros(n_paths)
    log_weight = np.zeros(n_paths)  # log Radon-Nikodym derivative
    overflow = np.zeros(n_paths, dtype=bool)

    for k in range(n_steps):
        h_k = h_star[k]
        noise_orig = rng.normal(0.0, 1.0, n_paths)  # standard normal
        # Under IS measure: Z_k = Z_k_orig + h_k * sqrt_dt
        noise_is = noise_orig + h_k * sqrt_dt
        Q = np.maximum(0.0, Q + d * dt + sigma * sqrt_dt * noise_is)
        overflow |= (Q >= B)
        # Log Radon-Nikodym: d ln W = -h_k * dW - h_k^2/2 * dt
        log_weight += -h_k * sqrt_dt * noise_is - 0.5 * h_k**2 * dt

    # IS estimator: E[1_{overflow} * W]
    weights = np.exp(log_weight)
    indicator = overflow.astype(float)
    prob_is = float(np.mean(indicator * weights))

    # Variance of IS estimator
    var_is = float(np.var(indicator * weights) / n_paths)
    se_is = float(np.sqrt(max(var_is, 0.0)))

    return prob_is, se_is, best_I


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def run_validation(lam, mu, T, dt, B_values, n_paths_direct, n_paths_is, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    print(f"{'B':>6} {'log P_direct':>14} {'log P_IS':>12} {'log P_theory':>14} {'gap':>8}")
    print("-" * 60)

    for B in B_values:
        t0 = time.perf_counter()

        # Direct MC (expensive for large B)
        prob_direct, se_direct = direct_mc_overflow_prob(
            lam, mu, B, T, n_paths_direct, dt, seed)
        log_p_direct = float(np.log(prob_direct)) if prob_direct > 1e-10 else -np.inf

        # IS MC
        prob_is, se_is, I_opt = importance_sampled_mc(
            lam, mu, B, T, n_paths_is, dt, seed + 1)
        log_p_is = float(np.log(prob_is)) if prob_is > 1e-10 else -np.inf

        # Theoretical rate function
        log_p_theory = float(-I_opt)

        gap = abs(log_p_is - log_p_theory) if np.isfinite(log_p_is) else np.nan
        elapsed = time.perf_counter() - t0

        print(f"{B:6.1f} {log_p_direct:14.3f} {log_p_is:12.3f} {log_p_theory:14.3f} {gap:8.3f}  ({elapsed:.1f}s)")

        results.append({
            "B": B,
            "log_p_direct": log_p_direct,
            "log_p_is": log_p_is,
            "log_p_theory": log_p_theory,
            "rate_function_I": I_opt,
            "gap_is_vs_theory": gap,
            "prob_direct": prob_direct,
            "prob_is": prob_is,
            "se_direct": se_direct,
            "se_is": se_is,
        })

    out_path = os.path.join(out_dir, "rate_function_validation.json")
    with open(out_path, "w") as f:
        json.dump({"lam": lam, "mu": mu, "T": T, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot — publication quality
    try:
        import matplotlib
        matplotlib.rcParams.update({"font.size": 11, "axes.labelsize": 12})
        import matplotlib.pyplot as plt
        from scipy.stats import linregress

        Bs = np.array([r["B"] for r in results])
        lp_th = np.array([r["log_p_theory"] for r in results])
        lp_dir = np.array([r["log_p_direct"] for r in results])

        # Fit linear slope to direct-MC estimates (finite B values where P < 1)
        valid = np.isfinite(lp_dir) & (lp_dir < -0.05)
        slope, intercept, r2, _, _ = (None,) * 5
        if valid.sum() >= 2:
            slope, intercept, r_val, _, _ = linregress(Bs[valid], lp_dir[valid])
            r2 = r_val**2
            fit_line = slope * Bs + intercept

        fig, ax = plt.subplots(figsize=(6, 4))

        # Theory rate function
        theory_finite = np.isfinite(lp_th) & (lp_th < -0.01)
        if theory_finite.any():
            ax.plot(Bs[theory_finite], lp_th[theory_finite], "k--",
                    linewidth=2, label=r"Theory: $-I(\varphi^*)$")

        # Direct MC scatter
        ax.scatter(Bs, lp_dir, color="tab:blue", s=50, zorder=5,
                   label=r"Direct MC $\hat{P}$")

        # Linear fit
        if valid.sum() >= 2:
            ax.plot(Bs, fit_line, color="tab:blue", linestyle=":",
                    label=f"MC fit slope={slope:.3f}")

        ax.set_xlabel(r"Overflow level $B$")
        ax.set_ylabel(r"$\log \hat{P}(Q_{\max} \geq B)$")
        ax.set_title(r"Rate-Function Validation: TV-$\sigma$ SDE")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(out_dir, "rate_function_validation.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {png_path}")
        if valid.sum() >= 2:
            print(f"  MC slope={slope:.3f}  (theory slope from I(φ*)={lp_th[valid][-1]/Bs[valid][-1]:.3f} per unit B)")
    except ImportError:
        print("matplotlib not available — skipping plot")

    return results


def main():
    parser = argparse.ArgumentParser(description="Rare-event rate-function validation")
    parser.add_argument("--lam", type=float, default=1.2, help="Arrival rate λ")
    parser.add_argument("--mu", type=float, default=1.0, help="Service rate μ")
    parser.add_argument("--T", type=float, default=10.0, help="Time horizon")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step")
    parser.add_argument("--B-values", type=float, nargs="+",
                        default=[1.0, 2.0, 3.0, 4.0, 5.0],
                        help="Overflow levels B to test")
    parser.add_argument("--n-paths-direct", type=int, default=50000,
                        help="Paths for direct MC")
    parser.add_argument("--n-paths-is", type=int, default=10000,
                        help="Paths for importance-sampled MC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/rare_event")
    args = parser.parse_args()

    print(f"TV-σ SDE: λ={args.lam}, μ={args.mu}, T={args.T}, dt={args.dt}")
    print(f"B values: {args.B_values}")
    print()

    run_validation(
        lam=args.lam, mu=args.mu, T=args.T, dt=args.dt,
        B_values=args.B_values,
        n_paths_direct=args.n_paths_direct,
        n_paths_is=args.n_paths_is,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
