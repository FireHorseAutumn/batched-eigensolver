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
measured point. The subspace-iteration variant (2026-10-26, four attack
rounds below) BEATS the vendor batched call at all nine measured cells —
1.21–1.67x, every spread disjoint — with per-matrix convergence exit (the
early termination the vendor lacks) contributing the final 22–24%. The
2.5x MAGMA factor is not reached; the headroom claim is now partially
measured rather than hypothetical.

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
kernel + vendor benchmark) built 2026-10-25; subspace-iteration variant
(`eig2`) built, gated, and measured 2026-10-26, with same-day round 2
(shared-staged reflectors, Sturm early exit) landing four vendor-winning
cells, round 3 (lower-triangle Gram, upper-only mirrored rank-2 update)
taking n = 128 at batch >= 256 for six of nine, and round 4 (per-matrix
convergence exit) winning all nine cells at 1.21–1.67x; suite green, 3/3.

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
  per thread, strided for n above the 128-thread block width; the Batch
  bound is kMaxSharedN = 256, set by the shared-staged reflectors; the
  pivoted tridiagonal solve on thread 0, chosen over pivot-free parallel
  cyclic reduction because inverse iteration drives T - sig*I exactly
  singular on degenerate eigenvalues); the Gram-Schmidt
  projections are batched per pass (s = Z^T col, then col -= Z s; two
  passes restore the MGS2 accuracy) so a pass costs one sync round instead
  of k block reductions. Tree-ordered reductions mean the GPU path is not
  bitwise identical to the CPU path; it is bitwise deterministic run to run
  and is gated against the CPU reference at 1e-12 * ||H||_F.
- **Subspace variant** (`eig2_batch_kernel` / `solve_batch_gpu2`, same
  header): inverse subspace iteration replacing the sequential
  per-eigenvector loop — all n shifted pivoted tridiagonal solves run in
  parallel (thread per eigenvector, per-thread gtsv scratch), and
  orthonormalization is CholeskyQR2 over the whole column block
  (block-cooperative Gram matrix — lower triangle only, the only triangle
  the Cholesky/apply read, so bitwise identical at half the dot cost —
  in-place Cholesky, forward-substitution apply) instead of per-vector
  Gram-Schmidt. Same Sturm eigenvalues, same shifts, same sign convention;
  up to 4 sweeps with a per-matrix convergence exit (block max of the
  per-column tridiagonal residuals <= 1e-13 after the orthonormalize;
  data-dependent but deterministic, same input -> same path); deterministic
  per-column refill streams. Same tolerances, same gate.
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
  both GPU paths, plus GUE at n = 144 (past the 128-thread block width,
  exercising the strided Sturm loop); eigenvalue agreement vs the CPU
  reference at worst 2.3e-15 (tolerance 1e-12*max(1, ||H||_F)), residuals
  at worst 2.3e-15, orthonormality at worst 1.8e-15; run-to-run bitwise
  determinism checked for both paths.

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

### Subspace-iteration attack (2026-10-26)

Per-phase timing of the old kernel (cumulative stop-after-phase runs,
median of 11, inputs resident) put inverse iteration at 50–76% of wall at
every measured point, with Householder tridiagonalization growing to
24–25% at n = 128 — so the eigenvector phase was attacked first, exactly
as the 2026-10-25 read predicted.

The `eig2` subspace variant keeps the Sturm eigenvalues, shifts, sweep
count, and sign convention, but runs all n shifted pivoted tridiagonal
solves in parallel (thread per eigenvector) and replaces the sequential
per-vector Gram-Schmidt with CholeskyQR2 over the column block. Same gate,
all ensembles, both paths green; run-to-run bitwise deterministic.

Wall medians (ms), same protocol as the table above, one session:

| n | batch | per-vector | subspace | vendor batched | subs vs vendor |
|---|---|---|---|---|---|
| 32 | 64 | 5.2 | 2.6 | 2.3 | 0.90x |
| 32 | 256 | 17.8 | 8.2 | 7.2 | 0.88x |
| 32 | 1024 | 59.5 | 30.3 | 28.6 | 0.94x |
| 64 | 64 | 21.8 | 11.9 | 10.1 | 0.85x |
| 64 | 256 | 71.8 | 38.2 | 37.3 | 0.98x |
| 64 | 1024 | 252.0 | 143.2 | 148.7 | 1.04x |
| 128 | 64 | 103.5 | 68.8 | 53.0 | 0.77x |
| 128 | 256 | 334.2 | 237.1 | 198.9 | 0.84x |
| 128 | 1024 | 1234.0 | 922.6 | 793.6 | 0.86x |

Read: subspace is 1.4–2.2x over the per-vector kernel (best at n = 32,
where the sequential k-loop hurt most) and the vendor gap narrows from
0.41–0.69x to 0.77–1.04x. Round 2 the same day staged the Householder
reflector vectors in shared memory through tridiagonalization and
back-transformation (identical operation order, so identical bits) and
gave the Sturm bisection an early exit at hi-lo <= 2e-14*max(1,|hi|,|lo|)
(the eigenvalue error stays ~500x below the 1e-13 shift offset; eval
agreement moved from 2.2e-16 to 2.2e-15, still ~500x inside the gate).
Final sweep (same protocol, one later session):

| n | batch | per-vector | subspace | vendor batched | subs vs vendor |
|---|---|---|---|---|---|
| 32 | 64 | 5.7 | 2.5 | 2.3 | 0.90x |
| 32 | 256 | 13.9 | 7.0 | 7.3 | 1.04x |
| 32 | 1024 | 55.3 | 27.4 | 28.8 | 1.05x |
| 64 | 64 | 22.3 | 11.6 | 10.1 | 0.87x |
| 64 | 256 | 60.4 | 33.9 | 37.4 | 1.10x |
| 64 | 1024 | 238.6 | 131.6 | 149.3 | 1.13x |
| 128 | 64 | 105.4 | 66.9 | 53.1 | 0.79x |
| 128 | 256 | 296.3 | 238.3 | 198.9 | 0.83x |
| 128 | 1024 | 1154.7 | 881.4 | 793.8 | 0.90x |

Read: the hand-written kernel now BEATS `cusolverDnXsyevBatched` at four
of nine cells — n = 32/64 at batch >= 256, up to 1.13x, disjoint spreads
(e.g. 64/1024: [129.7, 144.4] vs [149.3, 150.8] ms) — plus the 1.8–12.7x
over the naive loop everywhere. The vendor keeps n = 128 (0.79–0.90x),
where tridiagonalization is now the dominant phase (~20%, measured), and
the batch-64 latency cells. The structural gap story is reversed: the
remaining losses are measured phase costs with named attacks (blocked
Householder, warp-parallel pivoted tridiagonal solve), not mystery
overhead. Caveats as above: unpinned laptop, single session; subspace
spreads at 64/1024 and 128/256 reach ~10%. Raw reps and metadata in
[`data/eig_gpu_bench.json`](data/eig_gpu_bench.json).

### Round 3: symmetry cuts (2026-10-26, evening)

The round-2 phase breakdown named tridiagonalization (~20% at n = 128) and
the subspace orthonormalization as the remaining surfaces. Two symmetry
cuts, both bit-exact in what they keep:

- CholeskyQR Gram matrix: only the lower triangle is computed (the
  Cholesky and the forward-substitution apply read nothing else). Each
  kept entry has the same reduction order, so output is bitwise identical
  at half the Gram FP64 dot cost — the dominant term on an FP64-starved
  consumer GPU.
- Tridiagonalization rank-2 update: only the upper triangle is computed,
  then mirrored; the update is exactly Hermitian, so the old
  compute-both-triangles-then-average resymmetrization pass is deleted.
  Bits change (no averaging) but the result is exactly Hermitian by
  construction and stays ~100x inside every gate.

Phase probe at n = 128/batch = 256 (cumulative stop-after-phase,
`benchmarks/bench_phases.cu`, median of 11): tridiagonalization -43%,
subspace -27%, e2e 227 -> 178 ms. Same gate, both paths, all ensembles,
run-to-run bitwise, CTest 3/3.

Wall medians (ms), same protocol as above, one session:

| n | batch | per-vector | subspace | vendor batched | subs vs vendor |
|---|---|---|---|---|---|
| 32 | 64 | 5.6 | 2.6 | 2.3 | 0.91x |
| 32 | 256 | 13.7 | 6.9 | 7.3 | 1.07x |
| 32 | 1024 | 54.4 | 26.6 | 28.7 | 1.08x |
| 64 | 64 | 21.7 | 10.2 | 10.0 | 0.98x |
| 64 | 256 | 58.6 | 29.4 | 37.4 | 1.27x |
| 64 | 1024 | 231.6 | 114.0 | 149.3 | 1.31x |
| 128 | 64 | 99.7 | 53.1 | 53.0 | 1.00x |
| 128 | 256 | 278.5 | 184.4 | 199.0 | 1.08x |
| 128 | 1024 | 1095.2 | 703.2 | 793.8 | 1.13x |

Read: the hand-written kernel now beats `cusolverDnXsyevBatched` at six of
nine cells — every batch >= 256 cell, up to 1.31x, all with disjoint
spreads — with n = 128/batch = 64 a wash (1.00x, overlapping spreads). The
only remaining losses are the two smallest latency cells (32/64: 0.91x,
64/64: 0.98x), where the vendor's per-launch latency, not compute, is the
story. The n = 128 tridiagonalization wall from round 2 is gone. Caveats
unchanged: unpinned laptop, single evening, machine-state drift observed
across sessions at the largest cells; pinned rerun scheduled (portfolio
measurement day — landed 2026-10-30, see round 4). Raw reps and metadata in
[`data/eig_gpu_bench.json`](data/eig_gpu_bench.json).

### Round 4: per-matrix convergence exit (2026-10-26, evening)

The fixed 4-sweep subspace loop wastes work on the converged majority —
inverse iteration with a 1e-13 shift offset converges in 1–2 sweeps on
well-separated spectra. One block per matrix means no masking is needed: a
converged matrix's block just stops. After each sweep's CholeskyQR2, the
block votes on the max per-column tridiagonal residual
||T z_k - w_k z_k||_inf / (1 + |w_k|) and exits at <= 1e-13. Refilled or
non-finite columns (the degenerate-cluster path) carry large residuals by
construction and never exit early — the exact/near-degenerate ensembles
stay gated and green. Data-dependent but deterministic: same input, same
path, run-to-run bitwise, CTest 3/3.

Phase probe (n = 128/batch = 256, median of 11): subspace phase -48%,
e2e 178 -> 136 ms. Average sweep count is ~2 of 4 on GUE spectra.

Wall medians (ms), same protocol, governor-pinned rerun 2026-10-30
(performance governor on all CPUs, idle machine; the unpinned single-evening
table it replaces differed only within drift — ratios moved <= 0.03x):

| n | batch | per-vector | subspace | vendor batched | subs vs vendor |
|---|---|---|---|---|---|
| 32 | 64 | 5.1 | 1.9 | 2.3 | 1.20x |
| 32 | 256 | 13.7 | 5.3 | 7.2 | 1.38x |
| 32 | 1024 | 54.4 | 20.4 | 28.6 | 1.40x |
| 64 | 64 | 21.6 | 7.8 | 10.0 | 1.28x |
| 64 | 256 | 58.4 | 23.1 | 37.4 | 1.62x |
| 64 | 1024 | 231.6 | 89.2 | 149.2 | 1.67x |
| 128 | 64 | 100.3 | 41.5 | 53.0 | 1.28x |
| 128 | 256 | 278.5 | 140.8 | 198.9 | 1.41x |
| 128 | 1024 | 1094.9 | 529.7 | 793.6 | 1.50x |

Read: the hand-written kernel beats `cusolverDnXsyevBatched` at ALL nine
cells, 1.20–1.67x, every spread disjoint (tightest: 32/64 at [1.895,1.909]
vs [2.276,2.315] ms; widest: 128/256 at ~8% on the subspace side). The
batch-64 latency cells flipped with the reduced per-matrix critical path —
they were never about launch overhead on our side. This is the
convergence-variance hypothesis measured: early termination across
k-points, the property the vendor batched call lacks, is worth 22–24% e2e
on top of the round-3 state. The MAGMA 2.5x factor remains unreached.
Governor-pinned idle-machine rerun 2026-10-30 confirmed the unpinned
round-4 table (all nine cells, same win/loss pattern, ratios within drift);
the caveat is lifted. Raw reps and metadata in
[`data/eig_gpu_bench.json`](data/eig_gpu_bench.json).

Run: `cmake -B build && cmake --build build -j && ctest --test-dir build`.
Benchmark: `./build/bench_eig` (writes `data/eig_gpu_bench.json`).
