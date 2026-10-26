# batched-eigensolver

Batched diagonalization of many small (32–128) Hermitian matrices, one per
k-point of a tight-binding or Bogoliubov–de Gennes Hamiltonian.

## Vendor gap

cuSOLVER's batched eigensolver is generic: host overhead per batch element
and no early termination for converged problems. MAGMA's 2025 masked
early-exit result showed a 2.5x factor over cuSOLVER is available. As
measured here (see below), the straightforward block-per-matrix port does
NOT capture that headroom: it beats a naive per-matrix vendor loop
(1.8–12.7x) but trails `cusolverDnXsyevBatched` (0.41–0.69x) on every
measured point. The convergence-variance win remains a hypothesis, not a
result.

## Physics payload

The workhorse under every BdG calculation: topological, moiré, and
twisted-cuprate band structures and Majorana local-DOS calculations. Direct
extension of the tight-binding spine in
[kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding).

## Plan (all three steps landed)

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

Phase 1 (CPU reference + oracle harness) built 2026-10-23; phase 2 (GPU
kernel + vendor benchmark) built 2026-10-25; suite green, 3/3.

- **Layout:** `include/beig/hermitian_eig.hpp` — header-only,
  dependency-free FP64 solver. Per matrix: Householder reduction to real
  symmetric tridiagonal T (complex reflectors stored; a diagonal unitary D
  absorbs the off-diagonal phases), Sturm-sequence bisection per eigenvalue
  (Gershgorin bounds, fixed 80 iterations, `sp_core.h` pattern), shifted
  inverse iteration on T with pivoted tridiagonal solves (dgtsv recurrence)
  and full twice-modified Gram-Schmidt against all previously computed
  eigenvectors every sweep (fixed 4 sweeps; degenerate clusters are handled
  by the reorthogonalization, not by special casing), then
  back-transformation u = Q D z. Batch API: `solve_batch` over per-matrix
  (n, A, w, V) items; item b runs at seed `mix64(seed + b)`, bitwise
  reproducible.
- **GPU port** (`include/beig/hermitian_eig_gpu.cuh`): one thread block
  (128 threads) per matrix over a per-matrix slab of a global arena — at
  n = 128 the complex workspace alone is 528 KB, past any shared-memory
  budget. Sequential recurrences stay sequential (Sturm runs one eigenvalue
  per thread; the pivoted tridiagonal solve on thread 0, chosen over
  pivot-free parallel cyclic reduction because inverse iteration drives
  T - sig*I exactly singular on degenerate eigenvalues); the Gram-Schmidt
  projections are batched per pass (s = Z^T col, then col -= Z s; two
  passes restore the MGS2 accuracy) so a pass costs one sync round instead
  of k block reductions. Tree-ordered reductions mean the GPU path is not
  bitwise identical to the CPU path; it is bitwise deterministic run to run
  and is gated against the CPU reference at 1e-12 * ||H||_F.
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
- GPU gate (`tests/test_gpu_eig.cu`): the same ensembles batched through
  the GPU kernel; eigenvalue agreement vs the CPU reference at worst
  2.2e-16 (tolerance 1e-12*max(1, ||H||_F)), residuals at worst 9.4e-16,
  orthonormality at worst 1.8e-15; run-to-run bitwise determinism checked.

## Measured (2026-10-25)

NVIDIA RTX 5060 Laptop (sm_120, 26 SMs), nvcc 13.2.51 (g++-15 host), GUE
ensembles, median of 11 reps, wall = sync-bracketed host time; every case
gated GPU-vs-CPU at 1e-12*||H||_F before timing (worst observed 3.4e-16).
Raw per-rep wall and CUDA-event timings plus full device/toolchain metadata
in [`data/eig_gpu_bench.json`](data/eig_gpu_bench.json).

Wall medians (ms) and speedup ratios, beig vs `cusolverDnXsyevBatched`
(vendor batched) and a naive per-matrix `cusolverDnXsyevd` loop:

| n | batch | beig | vendor batched | naive loop | vs vendor | vs loop |
|---|---|---|---|---|---|---|
| 32 | 64 | 5.2 | 2.3 | 46.7 | 0.44x | 9.0x |
| 32 | 256 | 17.7 | 7.2 | 187.1 | 0.41x | 10.6x |
| 32 | 1024 | 59.0 | 28.4 | 751.3 | 0.48x | 12.7x |
| 64 | 64 | 21.7 | 10.0 | 147.6 | 0.46x | 6.8x |
| 64 | 256 | 89.7 | 39.2 | 616.2 | 0.44x | 6.9x |
| 64 | 1024 | 251.8 | 151.7 | 2721.7 | 0.60x | 10.8x |
| 128 | 64 | 112.7 | 57.6 | 339.4 | 0.51x | 3.0x |
| 128 | 256 | 391.0 | 245.1 | 1694.0 | 0.63x | 4.3x |
| 128 | 1024 | 3662.8 | 2512.1 | 6451.2 | 0.69x | 1.8x |

Read: the hand-written kernel beats the per-matrix vendor loop everywhere
(1.8–12.7x) but loses to the vendor's batched call at every point measured
(0.41–0.69x), with the gap narrowing as n grows. The MAGMA-style headroom
the vendor-gap section motivates was NOT captured by this straightforward
block-per-matrix port; the remaining gap is structural (the per-matrix
critical path is the sequential four-sweep inverse iteration with pivoted
thread-0 solves) and closing it is future work, not claimed here. Caveats:
unpinned laptop runs, so absolute times drift with machine state — the
n=128/batch=1024 row ran ~1.4x slower than an earlier same-evening
configuration while the vendor column drifted ~2.3x the other way; treat
ratios at the largest size as +/-50%. Vendor call reads the row-major slabs
as column-major conj(A), which has identical eigenvalues and the same work.

Run: `cmake -B build && cmake --build build -j && ctest --test-dir build`.
Benchmark: `./build/bench_eig` (writes `data/eig_gpu_bench.json`).
