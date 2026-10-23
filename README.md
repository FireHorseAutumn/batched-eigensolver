# batched-eigensolver

Batched diagonalization of many small (32–128) Hermitian matrices, one per
k-point of a tight-binding or Bogoliubov–de Gennes Hamiltonian.

## Vendor gap

cuSOLVER's batched eigensolver is generic: host overhead per batch element
and no early termination for converged problems. MAGMA's 2025 masked
early-exit result showed a 2.5x factor over cuSOLVER is available. The win
here comes from exploiting convergence variance across k-points, not from a
better dense eigensolver.

## Physics payload

The workhorse under every BdG calculation: topological, moiré, and
twisted-cuprate band structures and Majorana local-DOS calculations. Direct
extension of the tight-binding spine in
[kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding).

## Plan

1. CPU reference: batched Sturm-sequence bisection plus inverse iteration,
   generalizing the pattern proven in
   [nanosheet-sp](https://github.com/FireHorseAutumn/nanosheet-sp/blob/main/include/sp_core.h).
   Oracle: Eigen `SelfAdjointEigenSolver` on random Hermitian ensembles and
   one real tight-binding k-grid.
2. GPU kernel: one thread block per matrix.
3. Benchmark against the cuSOLVER batched path and a naive per-matrix loop,
   across batch count and matrix size, with the KPM reporting discipline:
   medians, observed spreads, raw per-repetition timings preserved.

## Gates

- Per-matrix orthogonality and residual checks against the Eigen oracle at a
  stated tolerance before any benchmark exists.
- No speedup claim without raw repetitions and recorded machine state.
- Suite green before anything is called measured.

## Status

Phase 1 (CPU reference + oracle harness) built 2026-10-23; suite green.

- `include/beig/hermitian_eig.hpp` — header-only, dependency-free FP64
  solver. Per matrix: Householder reduction to real symmetric tridiagonal T
  (complex reflectors stored; a diagonal unitary D absorbs the off-diagonal
  phases), Sturm-sequence bisection per eigenvalue (Gershgorin bounds, fixed
  80 iterations, `sp_core.h` pattern), shifted inverse iteration on T with
  pivoted tridiagonal solves (dgtsv recurrence) and full twice-modified
  Gram-Schmidt against all previously computed eigenvectors every sweep
  (fixed 4 sweeps; degenerate clusters are handled by the
  reorthogonalization, not by special casing), then back-transformation
  u = Q D z. Batch API: `solve_batch` over per-matrix (n, A, w, V) items;
  item b runs at seed `mix64(seed + b)`, bitwise reproducible.
- Oracle harness (`tests/test_eig.cpp`, Eigen `SelfAdjointEigenSolver`):
  GUE matrices n = 32/64/100/128, exactly degenerate clustered ensembles
  (integer levels, multiplicity ~n/8) and near-degenerate ones (1e-8
  jitter) at n = 64/128, and zigzag graphene ribbons (n = 2W) at
  W = 16/32/64 over 7 k-points. Tolerances: eigenvalues 1e-12*max(1, ||H||_F),
  residuals 1e-12*max(1, ||H||_F), orthonormality 1e-11. Measured worst
  cases: eval 1.4e-15, residual 9.1e-16, orthonormality 1.8e-15 (all
  relative where stated) — ~100x inside tolerance.
- Physics checks (`tests/test_ribbon.cpp`): ribbon bond count and
  bit-exact Hermiticity; at k = pi the exact spectrum {0 x2 (zigzag edge
  modes), +/-|t| x (W-1)} is recovered (edge modes at 2e-24, dimer levels
  at 9e-16); chiral +/-E pairing at 9e-16; batch output bitwise equal to
  per-matrix solves.

Run: `cmake -B build && cmake --build build -j && ctest --test-dir build`.
