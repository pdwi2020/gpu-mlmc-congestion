# Cluster Computing (Springer) Submission Package

This directory contains the manuscript reformatted for **Cluster Computing**
(Springer, Q1, IF 4.1).

## Before compiling

1. Download the Springer `svjour3` LaTeX template from the Springer author support page:
   <https://www.springernature.com/gp/authors/campaigns/latex-author-support>

2. Extract and copy these two files into this directory:
   - `svjour3.cls`
   - `spbasic.bst`

## Compile

```bash
latexmk -pdf main.tex
```

## Files

| File | Purpose |
|---|---|
| `main.tex` | Main manuscript (svjour3 class) |
| `main.bib` | Bibliography |
| `figures/` | Symlink to `../figures/` |

## Changes from `gpuAcc.tex` (IEEE Access)

| Element | IEEE Access | Cluster Computing |
|---|---|---|
| Document class | `ieeeaccess` | `svjour3[smallextended]` |
| Author block | `\address` | `\institute` |
| Keywords | `\begin{keywords}` | `\keywords{...}` |
| Bibliography style | `IEEEtran` | `spbasic` |
| Wide tables | `table*` | `table` (single-column) |
