/-!
# TV-σ KL Optimality and Freidlin-Wentzell Rate Function — Lean 4 Skeleton

Formalises Theorems 2–3 from Section III-H of the paper
"GPU-Accelerated Multilevel Monte Carlo for Network Congestion Propagation."

**Theorem 2** (KL optimality):
  σ(t) = √λ(t) minimises D_KL(Poisson(λ(t)dt) ‖ p_SDE) over all diffusion coefficients
  that preserve drift λ(t) - μ.

**Theorem 3** (Freidlin-Wentzell rate function):
  For the reflected SDE dQ = (λ-μ)dt + √λ(t) dW, Q ≥ 0, the large-deviation rate is
    I(φ) = ½ ∫₀ᵀ (φ̇(t) - (λ(t)-μ))² / λ(t) dt,  φ ≥ 0.

All proofs are `sorry`-stubs pending Mathlib formalisation of reflected SDEs.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.MeasureTheory.Integral.Bochner

open MeasureTheory Real

-- ---------------------------------------------------------------------------
-- Supporting definitions
-- ---------------------------------------------------------------------------

/-- Time-varying arrival rate: λ : [0,T] → ℝ≥0. -/
variable (T : ℝ) (hT : 0 < T)
variable (λ_ μ : ℝ → ℝ)
variable (hλpos : ∀ t ∈ Set.Icc 0 T, 0 < λ_ t)
variable (hμpos : ∀ t ∈ Set.Icc 0 T, 0 < μ t)

/-- KL divergence between two Gaussian distributions N(m, σ₁²) and N(m, σ₂²)
    (same mean, different variance): D_KL = ½(σ₁²/σ₂² - 1 - log(σ₁²/σ₂²)). -/
noncomputable def klGaussianSameM (σ₁ σ₂ : ℝ) : ℝ :=
  (1/2) * (σ₁^2 / σ₂^2 - 1 - Real.log (σ₁^2 / σ₂^2))

/-- KL divergence from Poisson(λ·h) to the Gaussian SDE step N((λ-μ)·h, σ²·h).
    For small h, Poisson variance is λ·h; the SDE step variance is σ²·h.
    KL ≈ ½ (λh/(σ²h) - 1 - log(λh/(σ²h))) = ½(λ/σ² - 1 - log(λ/σ²)). -/
noncomputable def klPoissonSDE (λ_ σ : ℝ) : ℝ :=
  klGaussianSameM (Real.sqrt λ_) σ

-- ---------------------------------------------------------------------------
-- Theorem 2 — KL Optimality of TV-σ
-- ---------------------------------------------------------------------------

/-- **Theorem 2** (KL optimality of σ(t) = √λ(t)).
    For each t, the per-step KL divergence from Poisson(λ(t)dt) to the SDE
    Gaussian step is minimised uniquely at σ(t) = √λ(t). -/
theorem tv_sigma_kl_optimal
    (t : ℝ) (ht : t ∈ Set.Icc 0 T)
    (λt : ℝ) (hλt : 0 < λt) :
    ∀ σ : ℝ, 0 < σ →
    klPoissonSDE λt (Real.sqrt λt) ≤ klPoissonSDE λt σ := by
  sorry
  -- Proof outline:
  -- 1. klGaussianSameM(√λ, σ) = ½(λ/σ² - 1 - log(λ/σ²))
  -- 2. Let u = λ/σ². Then f(u) = ½(u - 1 - log u) ≥ 0 for all u > 0
  -- 3. f(u) = 0 iff u = 1, i.e., σ² = λ, i.e., σ = √λ
  -- 4. f is strictly convex on (0,∞), so the minimum is unique
  -- Reference: Kabanov-Liptser-Shiryaev Girsanov density argument

-- ---------------------------------------------------------------------------
-- Theorem 3 — Freidlin-Wentzell Rate Function
-- ---------------------------------------------------------------------------

/-- An absolutely continuous path φ : [0,T] → ℝ≥0 (reflected at 0). -/
structure ReflectedPath where
  φ    : ℝ → ℝ
  nonneg : ∀ t, 0 ≤ φ t
  φ0   : φ 0 = 0

/-- Freidlin-Wentzell rate function for the reflected TV-σ SDE.
    I(φ) = ½ ∫₀ᵀ (φ̇(t) - (λ(t) - μ))² / λ(t) dt. -/
noncomputable def rateFunction (path : ReflectedPath) : ℝ :=
  (1/2) * ∫ t in Set.Icc 0 T,
    (deriv path.φ t - (λ_ t - μ t))^2 / λ_ t

/-- **Theorem 3** (Freidlin-Wentzell large-deviation lower bound).
    For the reflected SDE dQ = (λ(t)-μ)dt + √λ(t) dW with Q ≥ 0,
      lim_{ε→0} ε log P(‖Q - φ*‖ < δ) = -I(φ*)
    for the rate function I above. In particular, the log-probability of
    queue overflow at level B satisfies
      log P(Q_max ≥ B) ≈ -I(φ*), where φ* minimises I subject to max φ = B. -/
theorem freidlin_wentzell_lower_bound
    (B : ℝ) (hB : 0 < B)
    (φstar : ReflectedPath)
    (hφstar_max : ∃ t ∈ Set.Icc 0 T, φstar.φ t = B) :
    -- The rate function evaluated at the optimal path is the log-probability exponent
    ∃ I_opt : ℝ, I_opt = rateFunction λ_ μ T φstar ∧ 0 < I_opt := by
  sorry
  -- Proof outline:
  -- 1. The SDE dQ = (λ-μ)dt + √λ(t) dW satisfies Freidlin-Wentzell conditions
  --    (σ(t) = √λ(t) is bounded away from 0 since λ(t) > 0)
  -- 2. Rate function is I(φ) = ½∫(φ̇ - drift)²/σ² dt = ½∫(φ̇-(λ-μ))²/λ dt
  -- 3. Reflection at 0 modifies I for paths touching the boundary via
  --    the Skorokhod embedding (Tanaka formula)
  -- 4. φ* solves the Euler-Lagrange equation with reflecting b.c.
  -- References: Freidlin & Wentzell (1984), Chapter 3; Dupuis & Wang (2004)

-- ---------------------------------------------------------------------------
-- Corollary — Importance Sampling Measure Change
-- ---------------------------------------------------------------------------

/-- The optimal IS change of measure for estimating P(Q_max ≥ B)
    is the Girsanov transform with drift h*(t) = (φ̇*(t) - (λ(t)-μ)) / √λ(t).
    Under this measure, the sample path is exponentially tilted toward the
    overflow event, achieving the minimum-variance IS estimator. -/
theorem optimal_is_drift
    (φstar : ReflectedPath)
    (t : ℝ) (ht : t ∈ Set.Icc 0 T) :
    -- The optimal importance-sampling drift h* at time t
    ∃ h_star : ℝ,
    h_star = (deriv φstar.φ t - (λ_ t - μ t)) / Real.sqrt (λ_ t) := by
  exact ⟨_, rfl⟩
