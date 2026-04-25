# Cover Letter — GPU-Accelerated MLMC for Network Congestion Propagation

**Manuscript:** *GPU-Accelerated Multilevel Monte Carlo for Network Congestion Propagation: Coupled SDE Models with Uncertainty Quantification*

**Author:** Paritosh Dwivedi (Vellore Institute of Technology, India)

**Target venues (in order of preference):**
1. *Journal of Network and Computer Applications* (Elsevier, JNCA)
2. *ACM Transactions on Modeling and Computer Simulation* (TOMACS)

---

## Why this work fits the venue

Network operators today must provision capacity under inherently stochastic
demand while keeping delay SLAs credible during bursts, failures, and diurnal
load changes. Deterministic planning and ad-hoc safety margins either waste
capacity or hide violation risk. Accurate uncertainty quantification (UQ) for
queueing networks is therefore an operational requirement, but classical Monte
Carlo (MC) simulation is computationally prohibitive at the scale and time
horizons that real planning loops demand.

This paper delivers (i) a novel sample-allocation rule, *Adaptive Network-Aware
MLMC* (ANA-MLMC), that minimises a centrality-, pilot-variance-, and
SLA-priority-weighted estimator variance and reduces exactly to standard Giles
allocation when weights are uniform; (ii) a coupled stochastic differential
equation model for network-wide congestion propagation that captures correlated
queue dynamics across the adjacency matrix; (iii) a GPU-MLMC implementation
that achieves up to 12.91× runtime speedup and 257.72× reduction in
computational work over single-level GPU-MC at $\varepsilon=0.01$; and (iv) an
end-to-end real-trace validation pipeline driven by MAWI sample-point-F
backbone data. Together these advance the state of the art in **network
uncertainty quantification at scale** — a topic squarely within the JNCA and
TOMACS scope.

---

## Response to prior reviewer feedback

The submitted version is a substantial revision in response to detailed
reviewer comments (Dr. Anindita Kundu, SCOPE, VIT). The reviewer's three
structural concerns and how each is addressed:

| Reviewer concern | Section now containing the response |
|---|---|
| Novelty: paper *applies* standard MLMC; needs a methodological innovation | **§III-F Adaptive Network-Aware MLMC**: derivation of the weighted Giles allocation $N_l^\star \propto \sqrt{V_l^{w}/C_l}$ with $V_l^{w} = \sum_i w_i V_{l,i}$ and weight vector $w = \mathrm{norm}(\gamma_C c_i + \gamma_V v_i^{\text{pilot}} + \gamma_S s_i)$. Consistency, complexity preservation, regression test against standard Giles. **§V-C Empirical comparison** (5 seeds × 4 ε targets on Barabasi-Albert $n=500$, NVIDIA T4) with Table VII and Fig. 4. |
| Validation: M/M/1 comparison was flawed (regimes mismatched, $37\times$ discrepancy footnote); CAIDA was static; no real traffic traces | **§V-B**: replaced the flawed M/M/1 mean comparison with a Heavy-Traffic Diffusion Sanity Check showing the reflected SDE recovers the analytical M/M/1 mean with relative error decreasing monotonically from 15.0% at $\rho=0.7$ to 6.5% at $\rho=0.9$, exactly as the heavy-traffic theorem predicts. **§V-D**: dynamic CAIDA topology with 24-h diurnal load, 15 burst events, and 25 link failures (Fig. 6). **§V-G**: real-MAWI sample-point-F backbone trace case study (3.91 M packets, 100 MB Range-limited download), with a deeper discrete-event Poisson G/D/1 cross-validation that honestly reports the SDE's diffusion-approximation envelope. |
| Presentation, related work, limitations, reproducibility | Abstract reframed (ISP-first, equations moved to methodology). Introduction paragraphs 1–2 rewritten. **Table I** extended with four prior-work rows and two new columns (Network-aware Sampling, Tail-risk UQ). **§V-H Limitations** upgraded to a six-paragraph standalone subsection covering diffusion regime, topology coupling, real-trace depth, network scale, ANA weight sensitivity, and CPU baseline. **Reproducibility:** a public artifact at <https://github.com/pdwi2020/gpu-mlmc-congestion> contains a one-command Dockerfile, `scripts/reproduce_all.sh` end-to-end runner, GitHub Actions CI, and the canonical experiment outputs (`results/mm1_sanity.json`, `results/ana_mlmc/ana_mlmc_results.csv`, `results/dynamic_topology/summary.json`, `results/real_trace_deep/deep_validation_summary.json`). |

---

## Summary of contributions versus state of the art

1. **ANA-MLMC** is, to our knowledge, the first MLMC sample-allocation rule
   that integrates graph centrality and SLA priority with the standard
   variance-and-cost trade-off, while preserving the canonical
   $\mathcal{O}(\varepsilon^{-2})$ complexity guarantee.
2. The **coupled propagation SDE** is integrated end-to-end with GPU-MLMC,
   including correlated CI bands across neighbouring nodes — addressing the
   per-node-independence gap of prior diffusion-approximation work.
3. The **MAWI real-trace pipeline** is, to our knowledge, the first published
   demonstration of MLMC-driven UQ on the public MAWI backbone trace, with an
   honest discrete-event reference cross-validation.
4. The **GPU-MLMC implementation** sustains $>$65,000 samples/s on 100-node
   networks and 12,000 samples/s on 500-node coupled-propagation runs;
   $12.91\times$ runtime and $257.72\times$ work reduction over single-level
   GPU-MC at tight accuracy ($\varepsilon=0.01$) on NVIDIA A100.

---

## Suggested reviewers (avoiding conflicts)

- An expert in MLMC theory (Giles-school)
- An expert in stochastic queueing / heavy-traffic limits
- An expert in GPU-accelerated scientific simulation
- An expert in network measurement and traffic modelling

(Specific names left blank for editorial discretion.)

---

## Statements

- **Originality.** The manuscript has not been published elsewhere and is not
  under review at any other venue.
- **Author contributions.** The author developed the methodology, implemented
  all software, ran the experiments, and prepared the manuscript.
- **Funding / conflicts.** None to declare.
- **Data availability.** Source code, experiment scripts, and all canonical
  result artifacts are publicly available (de-anonymised on acceptance) at the
  artifact URL above. The MAWI traces used are public; the licence terms of
  CAIDA AS-rel2 are followed.
- **AI-assisted writing disclosure.** Drafting and orchestration assistance
  was provided by an AI coding agent. All scientific claims, equations, and
  reported numbers were verified by the author against the committed source
  code and the canonical result JSON files.

I look forward to the editorial decision and the reviewers' comments.

Sincerely,

**Paritosh Dwivedi**
Vellore Institute of Technology
paritosh.dwivedi2024@vitstudent.ac.in
