"""
Validate Proposition (Local Variance Matching of TV-sigma):
  Gaussian N(lambda*dt, sigma^2*dt) matches mean+variance of Poisson(lambda*dt)
  iff sigma^2 = lambda.

The paper's proposition (Section III) uses the random-time-change representation
(Ethier & Kurtz 1986): under Poisson arrival, N(t) - lambda*t = sigma*W(t) + o(sqrt(t))
with sigma = sqrt(lambda) as the unique consistent choice.  This script validates:

  1. Moment matching: sigma^2=lambda is the UNIQUE choice that matches both the
     mean and variance of Poisson(lambda*dt) (analytically exact).

  2. W1 empirical check: W1(Poisson(lambda*dt), N(lambda*dt, sigma^2*dt)) is
     computed for a grid of sigma^2 values.  Note that for finite dt, W1 is
     minimised at sigma^2 < lambda because the Poisson atom at 0 (mass ~1-lambda*dt)
     favors a tightly concentrated Normal; only the moment-matching criterion
     (sigma^2=lambda) yields the canonical diffusion approximation.
     The paper does NOT claim W1 minimisation — it cites random-time-change.

Saves results to results/theory/wasserstein_sigma_validation.json.
"""

import json
import os
import numpy as np


def empirical_w1(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """W1 via sorted-sample formula (exact for 1-D empirical distributions)."""
    return float(np.mean(np.abs(np.sort(samples_a) - np.sort(samples_b))))


def validate():
    lambdas = [0.5, 1.0, 2.0, 5.0]
    dts = [0.1, 0.01, 0.001]
    sigma_sq_factors = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    rng = np.random.default_rng(42)
    n = 300_000
    results = []
    moment_match_unique = True  # analytically guaranteed

    # ----------------------------------------------------------------
    # 1. Moment-matching uniqueness (analytical)
    # ----------------------------------------------------------------
    print("=== 1. Moment-matching uniqueness (analytical) ===")
    print("Claim: sigma^2=lambda is the UNIQUE sigma that satisfies")
    print("  E[N(lambda*dt, sigma^2*dt)] = E[Poisson(lambda*dt)] = lambda*dt  AND")
    print("  Var[N(lambda*dt, sigma^2*dt)] = Var[Poisson(lambda*dt)] = lambda*dt")
    print("Proof: mean match -> mu=lambda*dt (trivial); var match -> sigma^2*dt=lambda*dt.")
    print("=> sigma^2 = lambda uniquely. Verified analytically: True\n")

    # ----------------------------------------------------------------
    # 2. W1 reference table (supplementary — paper does NOT claim W1 min)
    # ----------------------------------------------------------------
    print("=== 2. W1 reference (supplementary) ===")
    print("W1 = mean|sort(Poisson) - sort(Normal)| (empirical, n=300k).")
    print("NOTE: The paper uses the random-time-change justification,")
    print("not W1 minimisation.  W1 in finite dt favors sigma^2 < lambda")
    print("(smaller variance keeps Normal near Poisson atom at 0).\n")
    print(f"{'lambda':>8} {'dt':>8} {'W1 at sigma^2=lam':>20} "
          f"{'W1 at sigma^2=0.5lam':>22} {'W1 at sigma^2=0.25lam':>23}")
    print("-" * 87)

    for lam in lambdas:
        for dt in dts:
            mean_val = lam * dt
            w1_row = {}
            for factor in sigma_sq_factors:
                sigma_sq = factor * lam
                pois = rng.poisson(mean_val, size=n).astype(float)
                gauss = rng.normal(mean_val, np.sqrt(sigma_sq * dt), size=n)
                w1_row[factor] = empirical_w1(pois, gauss)

            print(f"{lam:>8.1f} {dt:>8.3f} {w1_row[1.0]:>20.6f} "
                  f"{w1_row[0.5]:>22.6f} {w1_row[0.25]:>23.6f}")

            results.append({
                "lambda": lam,
                "dt": dt,
                "poisson_mean": mean_val,
                "poisson_var": mean_val,
                "variance_match_holds": True,
                "w1_at_sigma_sq_eq_lambda": w1_row[1.0],
                "w1_at_sigma_sq_eq_0p5_lambda": w1_row[0.5],
                "w1_at_sigma_sq_eq_0p25_lambda": w1_row[0.25],
                "note": (
                    "sigma^2=lambda is the moment-matching choice (random-time-change). "
                    "W1 minimisation in finite dt favors smaller sigma (Poisson atom at 0)."
                ),
            })
        print()

    print("Conclusion:")
    print("  Moment matching (sigma^2=lambda): VERIFIED analytically (unique solution).")
    print("  W1 minimisation at sigma^2=lambda: NOT claimed by the paper.")
    print("  Paper justification: random-time-change representation (Ethier & Kurtz 1986).")

    out = {
        "proposition": (
            "sigma^2=lambda is the unique Gaussian variance that matches both "
            "the mean and variance of Poisson(lambda*dt). "
            "Justified by the random-time-change representation "
            "(Ethier & Kurtz 1986, not W1 minimisation)."
        ),
        "moment_match_verified": True,
        "w1_minimisation_claimed": False,
        "note": (
            "The email (item 1) suggested W1 minimisation as motivation; "
            "for finite dt W1 favors sigma^2 < lambda due to Poisson concentration. "
            "The paper's proposition correctly uses the random-time-change argument."
        ),
        "metric_used": "W1 empirical (sorted-sample), shown for reference only",
        "rows": results,
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "theory",
        "wasserstein_sigma_validation.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {os.path.abspath(out_path)}")
    return True


if __name__ == "__main__":
    validate()
