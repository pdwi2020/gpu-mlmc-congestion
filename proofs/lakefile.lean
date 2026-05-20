import Lake
open Lake DSL

package «ana_mlmc_proofs» where
  name := "ana_mlmc_proofs"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.14.0"

lean_lib «AnaMLMC» where
  roots := #[`AnaMLMCComplexity, `TVSigmaRateFunction]
