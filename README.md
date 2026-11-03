# BDF–FEM methods for the Hodgkin–Huxley model on branched axons

This repository contains the numerical code and selected data used in
Hayden Reed and Qingguang Guan, *High-Order Unconditionally Stable IMEX
BDF–FEM Schemes for the Hodgkin–Huxley Model on a Branched Axon*.
The paper is available on [arXiv (arXiv:2611.02423)](https://arxiv.org/abs/2611.02423).
The code implements the masked, linearly implicit BDF–FEM family for
orders 1–5 and the finite element degrees used in the paper (P1–P4).

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@misc{reed2026highorder,
  title         = {{High-Order Unconditionally Stable IMEX BDF--FEM Schemes for the Hodgkin--Huxley Model on a Branched Axon}},
  author        = {Reed, Hayden and Guan, Qingguang},
  year          = {2026},
  eprint        = {2611.02423},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  url           = {https://arxiv.org/abs/2611.02423}
}
```

## Setup

Use Python 3.10 or later. From the repository root, install the dependencies:

```bash
python -m pip install -r requirements-multiprecision.txt
```

The multiprecision experiment needs `mpmath`; use `requirements.txt` instead
if that experiment is not needed. The recorded runtime measurements used
Python 3.13.13, NumPy 2.4.4 and SciPy 1.17.1 on the hardware stated in the
paper. Timings depend on the machine and software environment.

Run the solver, multiprecision and benchmark tests with:

```bash
python -m unittest -v tests.test_solver tests.test_multiprecision tests.test_hh_runtime
```

## Experiments

Commands below run from the repository root. Simulations can take substantial
time, particularly the high-precision and 46 ms fine-grid experiments. Use
fresh output folders where supported to preserve the included results.

* **Manufactured profiles:** `python -m studies.manufactured.plot_profiles`
  plots the analytic fields in Figure 2.
* **Manufactured convergence (Tables 3–5):**
  `python -m studies.manufactured.run_joint_t1` runs the joint-refinement
  study in double precision. It also runs BDF5/P4 in double precision;
  the paper's Table 6 instead uses the 50-digit calculation below.
* **BDF5/P4 manufactured convergence (Table 6):**
  `python -m studies.manufactured.run_joint_t1_mp --dps 50` runs the
  multiprecision experiment. The default grids are 40, 80, 160 and 320
  elements per branch. The included CSV is the reported 50-digit result.
* **Two-junction graph (Figures 3–4):**
  `python -m studies.graph_geometry.verify_geometry --plot-only` redraws
  the figures from the included study report; omit `--plot-only` to
  recompute the experiment and its numerical controls.
* **Physical pulse (Figure 5):** `cd bdf3_p2_propagation` and run
  `python run_y_axon.py --no-plots` to rerun the BDF3/P2 pulse calculation.
  The saved fields used for the paper's figure are included under
  `outputs/y_axon/`. `plot_results.py` makes related propagation plots.
* **Physical convergence (Table 7 and Figure 6):**
  `studies/hh_convergence/` contains the physical solver and field
  comparison routines, the paper's saved candidate fields, and the common
  BDF5/P4 reference. The final fine-grid BDF3/P2 case is also included.
* **Work–precision comparison (Figure 7 and Table 8):**
  `python -m studies.hh_convergence.runtime_benchmark.run_benchmark --analyze-only`
  recomputes the summaries and figure from the 57 included timing records.
  For new timing runs, use `--output` with a new directory inside
  `studies/hh_convergence/runtime_benchmark/`; see `--help`. The complete
  benchmark runs each of 19 cases three times plus quadrature controls.
* **Mask (Figure 8):** `python plot_masking_function.py` draws the mask.

The saved numerical histories, reference and timing records are included
where needed to support the published experiments. The
work–precision errors use a finite numerical reference, as described in the
paper; the comparison concerns the specified joint-refinement paths.
