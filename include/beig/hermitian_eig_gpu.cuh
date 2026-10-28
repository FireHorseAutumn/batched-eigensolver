#pragma once

// hermitian_eig_gpu.cuh — GPU port of the beig solver, one thread block per
// matrix, FP64. Uniform n per launch; batch = gridDim.x blocks of
// BEIG_GPU_THREADS threads.
//
// The algorithm is the CPU one step for step: Householder reduction to real
// tridiagonal (complex reflectors + diagonal phase unitary), Sturm bisection
// per eigenvalue, shifted inverse iteration with pivoted tridiagonal solves
// and full twice-modified Gram-Schmidt, back-transformation u = Q D z.
// Sequential recurrences (Sturm chain, gtsv, phase accumulation) stay
// sequential — Sturm runs one eigenvalue per thread, strided past the block
// size (n may reach kMaxSharedN = 256 > kThreads), gtsv on thread 0; the
// O(n^2)/O(n^3) inner loops (norms, S*v, rank-2 update with mirroring,
// Gram-Schmidt projections, reflector application) are block-cooperative.
//
// Bit-level note: block reductions are tree-ordered, so GPU results are NOT
// bitwise identical to the CPU solver. Correctness gates are tolerance-based
// against Eigen and the CPU reference. The kernel itself is deterministic:
// fixed geometry, fixed reduction shape, same inputs -> same bits.

#include "hermitian_eig.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace beig {
namespace gpu {

constexpr int kThreads = 128;

// Reflector vectors (v, w, u) are staged in shared memory during
// tridiagonalization and back-transformation — the inner loops re-read them
// once per element otherwise. Bounds n; Batch construction throws past it.
constexpr int kMaxSharedN = 256;

// Complex double with std::complex<double> layout.
struct dc {
    double re, im;
};
static_assert(sizeof(dc) == sizeof(cdouble), "dc layout");

__host__ __device__ inline dc dadd(dc a, dc b) { return {a.re + b.re, a.im + b.im}; }
__host__ __device__ inline dc dsub(dc a, dc b) { return {a.re - b.re, a.im - b.im}; }
__host__ __device__ inline dc dmul(dc a, dc b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
__host__ __device__ inline dc dconj(dc a) { return {a.re, -a.im}; }
__host__ __device__ inline dc dscale(dc a, double s) { return {a.re * s, a.im * s}; }
__host__ __device__ inline double dabs(dc a) { return hypot(a.re, a.im); }

__device__ inline uint64_t mix64_d(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ inline double lcg_uniform_d(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(s >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

// Shared broadcast/reduction context. One per block.
struct ShCtx {
    double red[32];
    double scalar;   // general broadcast (nrm, mx, ss)
    double are, aim; // alpha
    int flag;
    int bad;
};

// Block-wide sum. All threads must call. red: shared, >= 32 doubles.
__device__ inline double block_sum(double v, ShCtx& sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) sh.red[warp] = v;
    __syncthreads();
    const int nwarp = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        v = (lane < nwarp) ? sh.red[lane] : 0.0;
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh.red[0] = v;
    }
    __syncthreads();
    const double r = sh.red[0];
    __syncthreads();
    return r;
}

// Block-wide max. All threads must call.
__device__ inline double block_max(double v, ShCtx& sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o; o >>= 1)
        v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) sh.red[warp] = v;
    __syncthreads();
    const int nwarp = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        v = (lane < nwarp) ? sh.red[lane] : -INFINITY;
        for (int o = 16; o; o >>= 1)
            v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
        if (lane == 0) sh.red[0] = v;
    }
    __syncthreads();
    const double r = sh.red[0];
    __syncthreads();
    return r;
}

__device__ inline int sturm_count_d(const double* d, const double* e, int n,
                                    double lam) {
    double q = d[0] - lam;
    int cnt = (q < 0.0) ? 1 : 0;
    for (int i = 1; i < n; ++i) {
        double qq = q;
        if (fabs(qq) < 1e-290) qq = (qq < 0.0) ? -1e-290 : 1e-290;
        q = (d[i] - lam) - e[i - 1] * e[i - 1] / qq;
        if (q < 0.0) ++cnt;
    }
    return cnt;
}

// Bisection as the CPU find_eigen, with an early exit at
// hi - lo <= 2e-14 * max(1, |hi|, |lo|): the eigenvalue error stays
// ~500x below the 1e-13 inverse-iteration shift offset, so the shift's
// ordering margin is untouched. Runs one eigenvalue per thread. Iteration
// count is data-dependent but deterministic (same input, same path).
__device__ inline double find_eigen_d(const double* d, const double* e, int n,
                                      int k) {
    double lo = d[0], hi = d[0];
    for (int i = 0; i < n; ++i) {
        double r = 0.0;
        if (i > 0) r += fabs(e[i - 1]);
        if (i < n - 1) r += fabs(e[i]);
        lo = fmin(lo, d[i] - r);
        hi = fmax(hi, d[i] + r);
    }
    lo -= 1.0;
    hi += 1.0;
    for (int it = 0; it < 80; ++it) {
        if (hi - lo <= 2e-14 * fmax(1.0, fmax(fabs(hi), fabs(lo)))) break;
        double mid = 0.5 * (lo + hi);
        if (sturm_count_d(d, e, n, mid) >= k + 1) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}

// Thread-0 port of gtsv_shift. s: 3n scratch. x may alias b.
__device__ inline void gtsv_shift_d(const double* d, const double* e,
                                    double sig, int n, const double* b,
                                    double* x, double* s) {
    double* dd = s;
    double* du = s + n;
    double* du2 = s + 2 * n;
    const double tiny = 1e-290;
    for (int i = 0; i < n; ++i) { dd[i] = d[i] - sig; x[i] = b[i]; }
    for (int i = 0; i < n - 1; ++i) { du[i] = e[i]; du2[i] = 0.0; }
    for (int i = 0; i < n - 1; ++i) {
        const double sub = e[i];
        if (fabs(dd[i]) >= fabs(sub)) {
            if (fabs(dd[i]) < tiny) dd[i] = (dd[i] < 0.0) ? -tiny : tiny;
            const double m = sub / dd[i];
            dd[i + 1] -= m * du[i];
            x[i + 1] -= m * x[i];
        } else {
            const double m = dd[i] / sub;
            dd[i] = sub;
            const double t = dd[i + 1];
            dd[i + 1] = du[i] - m * t;
            if (i + 1 < n - 1) {
                du2[i] = du[i + 1];
                du[i + 1] = -m * du2[i];
            }
            du[i] = t;
            const double tb = x[i];
            x[i] = x[i + 1];
            x[i + 1] = tb - m * x[i];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double p = dd[i];
        if (fabs(p) < tiny) p = (p < 0.0) ? -tiny : tiny;
        double r = x[i] - ((i + 1 < n) ? du[i] * x[i + 1] : 0.0)
                        - ((i + 2 < n) ? du2[i] * x[i + 2] : 0.0);
        x[i] = r / p;
    }
}

// Block-cooperative Householder reduction; mirrors tridiag_hermitian.
// The rank-2 update computes only the upper triangle and mirrors it (the
// update is exactly Hermitian; the diagonal is forced real there), so no
// separate re-symmetrization pass exists. v/w/u are shared-staged (the
// inner loops re-read them per element); vr still gets the persistent copy
// the back-transformation reads later.
__device__ inline void tridiag_hermitian_d(dc* a, int n, double* d, double* e,
                                           dc* vr, double* tau, double* phase,
                                           dc* cs, ShCtx& sh) {
    (void)cs; // superseded by the shared staging below
    __shared__ dc sv[kMaxSharedN];
    __shared__ dc sw[kMaxSharedN];
    __shared__ dc su[kMaxSharedN];
    const int B = blockDim.x, tid = threadIdx.x;
    for (int k = 0; k < n - 2; ++k) {
        const int m = n - k - 1;
        dc* vg = vr + (size_t)k * n + k + 1;
        double mine = 0.0;
        for (int i = tid; i < m; i += B) {
            const dc val = a[(size_t)(k + 1 + i) * n + k];
            vg[i] = val;
            sv[i] = val;
            mine += val.re * val.re + val.im * val.im;
        }
        const double nrm = sqrt(block_sum(mine, sh));
        if (nrm == 0.0) {
            if (tid == 0) {
                tau[k] = 0.0;
                a[(size_t)(k + 1) * n + k] = {0.0, 0.0};
                a[(size_t)k * n + k + 1] = {0.0, 0.0};
            }
            __syncthreads();
            continue;
        }
        if (tid == 0) {
            const dc x0 = sv[0];
            const double ax0 = dabs(x0);
            const dc ph = (ax0 > 0.0) ? dscale(x0, 1.0 / ax0) : dc{1.0, 0.0};
            const dc alpha = dscale(ph, -nrm);
            sh.are = alpha.re;
            sh.aim = alpha.im;
            tau[k] = 1.0 / (nrm * (nrm + ax0));
            sv[0] = dsub(x0, alpha);
            vg[0] = sv[0];
        }
        __syncthreads();
        const double tauk = tau[k];
        for (int p = tid; p < m; p += B) {
            const dc* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            dc acc = {0.0, 0.0};
            for (int q = 0; q < m; ++q) acc = dadd(acc, dmul(srow[q], sv[q]));
            sw[p] = acc;
        }
        __syncthreads();
        double mine2 = 0.0;
        for (int p = tid; p < m; p += B)
            mine2 += dmul(dconj(sv[p]), sw[p]).re;
        const double s2 = block_sum(mine2, sh);
        const double g = 0.5 * tauk * tauk * s2;
        for (int p = tid; p < m; p += B)
            su[p] = dsub(dscale(sw[p], tauk), dscale(sv[p], g));
        __syncthreads();
        // Rank-2 update, upper triangle only, mirrored: the update is
        // exactly Hermitian, so computing both triangles and averaging (the
        // old resymmetrization pass) buys nothing. Diagonal forced real.
        for (int p = tid; p < m; p += B) {
            dc* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            for (int q = p; q < m; ++q) {
                const dc upd =
                    dsub(srow[q],
                         dadd(dmul(sv[p], dconj(su[q])),
                              dmul(su[p], dconj(sv[q]))));
                srow[q] = (q == p) ? dc{upd.re, 0.0} : upd;
                if (q > p)
                    a[(size_t)(k + 1 + q) * n + k + 1 + p] = dconj(upd);
            }
        }
        __syncthreads();
        if (tid == 0) {
            a[(size_t)(k + 1) * n + k] = {sh.are, sh.aim};
            a[(size_t)k * n + k + 1] = {sh.are, -sh.aim};
            for (int i = k + 2; i < n; ++i) {
                a[(size_t)i * n + k] = {0.0, 0.0};
                a[(size_t)k * n + i] = {0.0, 0.0};
            }
        }
        __syncthreads();
    }
    for (int i = tid; i < n; i += B) d[i] = a[(size_t)i * n + i].re;
    for (int i = tid; i < n - 1; i += B)
        e[i] = dabs(a[(size_t)(i + 1) * n + i]);
    if (tid == 0) {
        if (n >= 2) tau[n - 2] = 0.0;
        tau[n - 1] = 0.0;
        phase[0] = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            const dc c = a[(size_t)(i + 1) * n + i];
            phase[i + 1] = phase[i] + atan2(c.im, c.re);
        }
    }
    __syncthreads();
}

// Block-cooperative Gram-Schmidt, two passes, against columns 0..k-1 of z;
// 2-normalize, refill from the LCG when depleted. st lives on thread 0.
// sc: n doubles of scratch (all projections of one pass). The projections
// are batched — s = Z^T col computed against the un-updated col (classical
// GS), then col -= Z s — so each pass costs one sync round instead of k
// block reductions; two passes restore the twice-modified-Gram-Schmidt
// accuracy of the CPU path. Per element the subtraction order over j is
// unchanged, but the projections see the original col, so GPU output is not
// bitwise identical to the CPU MGS2 path (tolerance-gated instead).
__device__ inline void orthonormalize_d(double* z, int k, int n, double* col,
                                        uint64_t& st, double* sc,
                                        ShCtx& sh) {
    const int B = blockDim.x, tid = threadIdx.x;
    for (int attempt = 0; attempt < 4; ++attempt) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = tid; j < k; j += B) {
                const double* zj = z + (size_t)j * n;
                double s = 0.0;
                for (int i = 0; i < n; ++i) s += zj[i] * col[i];
                sc[j] = s;
            }
            __syncthreads();
            for (int i = tid; i < n; i += B) {
                double acc = col[i];
                for (int j = 0; j < k; ++j) acc -= sc[j] * z[(size_t)j * n + i];
                col[i] = acc;
            }
            __syncthreads();
        }
        double mine = 0.0;
        for (int i = tid; i < n; i += B) mine += col[i] * col[i];
        const double ss = block_sum(mine, sh);
        if (ss > 1e-24) {
            const double inv = 1.0 / sqrt(ss);
            for (int i = tid; i < n; i += B) col[i] *= inv;
            __syncthreads();
            return;
        }
        if (tid == 0)
            for (int i = 0; i < n; ++i) col[i] = lcg_uniform_d(st);
        __syncthreads();
    }
}

// One eigenvector by shifted inverse iteration; mirrors
// eigvec_inverse_iter. The k-loop over eigenvectors stays sequential (each
// column reorthogonalizes against all previous columns); inner reductions
// are block-cooperative, gtsv runs on thread 0. ds: 4n scratch.
__device__ inline void eigvec_inverse_iter_d(const double* d, const double* e,
                                             int n, int k, double lam,
                                             uint64_t seed, double* z,
                                             double* ds, ShCtx& sh) {
    double* y = ds + 3 * n;
    double* col = z + (size_t)k * n;
    const double sig = lam + 1e-13 * (1.0 + fabs(lam));
    const int B = blockDim.x, tid = threadIdx.x;
    uint64_t st = seed;
    if (tid == 0)
        for (int i = 0; i < n; ++i) col[i] = lcg_uniform_d(st);
    __syncthreads();
    orthonormalize_d(z, k, n, col, st, ds, sh);
    for (int it = 0; it < 4; ++it) {
        if (tid == 0) gtsv_shift_d(d, e, sig, n, col, y, ds);
        __syncthreads();
        if (tid == 0) sh.bad = 0;
        __syncthreads();
        double mine = -1.0;
        for (int i = tid; i < n; i += B) {
            if (!isfinite(y[i])) sh.bad = 1;
            else mine = fmax(mine, fabs(y[i]));
        }
        const double mx = block_max(mine, sh);
        const int bad = sh.bad;
        if (bad || mx == 0.0) {
            if (tid == 0)
                for (int i = 0; i < n; ++i) col[i] = lcg_uniform_d(st);
        } else {
            const double inv = 1.0 / mx;
            for (int i = tid; i < n; i += B) col[i] = y[i] * inv;
        }
        __syncthreads();
        orthonormalize_d(z, k, n, col, st, ds, sh);
    }
    if (tid == 0) {
        int imx = 0;
        double mx = 0.0;
        for (int i = 0; i < n; ++i)
            if (fabs(col[i]) > mx) { mx = fabs(col[i]); imx = i; }
        sh.flag = (col[imx] < 0.0) ? 1 : 0;
    }
    __syncthreads();
    if (sh.flag)
        for (int i = tid; i < n; i += B) col[i] = -col[i];
    __syncthreads();
}

// out = Q D z, block-cooperative: D row-scaling flattened over all entries,
// then reflectors applied one column per thread, v shared-staged per step.
__device__ inline void back_transform_d(const dc* vr, const double* tau,
                                        const double* phase, const double* z,
                                        int n, dc* out) {
    __shared__ dc sv[kMaxSharedN];
    const int B = blockDim.x, tid = threadIdx.x;
    for (int idx = tid; idx < n * n; idx += B) {
        const int i = idx % n;
        double cr, sr;
        sincos(phase[i], &sr, &cr);
        out[idx] = {z[idx] * cr, z[idx] * sr};
    }
    __syncthreads();
    for (int k = n - 3; k >= 0; --k) {
        if (tau[k] == 0.0) continue;
        const int m = n - k - 1;
        const dc* vg = vr + (size_t)k * n + k + 1;
        for (int i = tid; i < m; i += B) sv[i] = vg[i];
        __syncthreads();
        for (int j = tid; j < n; j += B) {
            dc* yj = out + (size_t)j * n + k + 1;
            dc s = {0.0, 0.0};
            for (int i = 0; i < m; ++i)
                s = dadd(s, dmul(dconj(sv[i]), yj[i]));
            s = dscale(s, tau[k]);
            for (int i = 0; i < m; ++i)
                yj[i] = dsub(yj[i], dmul(s, sv[i]));
        }
        __syncthreads();
    }
}

// ---- Block subspace iteration variant ----------------------------------
//
// eig2 replaces the sequential per-eigenvector inverse iteration with
// inverse subspace iteration over the whole column block: all n shifted
// tridiagonal solves run in parallel (one thread per eigenvector, gtsv
// scratch per thread), and orthonormalization is CholeskyQR2 over the
// block (Gram matrix, in-place block-cooperative Cholesky, forward-
// substitution apply) instead of the sequential per-vector Gram-Schmidt.
// The measured motivation: per-phase timing shows the sequential k-loop
// with thread-0 gtsv at 50-76% of solver wall time.
//
// Determinism: fixed geometry, fixed reduction shapes, refill streams
// seeded per (matrix, column, sweep) — run-to-run bitwise like the
// per-vector path. Not bitwise vs the CPU solver (tree reductions,
// different orthonormalization) — the same tolerance gates apply.

// One CholeskyQR pass on the vector block z (vector k contiguous at
// z + k*n). G: n*n scratch (Gram, then in-place lower Cholesky factor).
// On diagonal breakdown sets sh.bad and returns early; the caller refills.
__device__ inline void cholqr_once_d(double* z, int n, double* G,
                                     ShCtx& sh) {
    const int B = blockDim.x, tid = threadIdx.x;
    // Lower triangle of the Gram matrix only: Cholesky and the apply below
    // read just i >= j, and each entry keeps its reduction order, so this is
    // bitwise identical to the full n^2 Gram at half the FP64 dot cost.
    for (int j = 0; j < n; ++j) {
        const double* zj = z + (size_t)j * n;
        for (int k = tid; k <= j; k += B) {
            const double* zk = z + (size_t)k * n;
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += zj[i] * zk[i];
            G[(size_t)j * n + k] = s;
        }
    }
    __syncthreads();
    if (tid == 0) sh.bad = 0;
    __syncthreads();
    // In-place lower Cholesky, one column per round. Every thread reads
    // t_j after the sync and computes sqrt redundantly — no broadcast.
    for (int j = 0; j < n; ++j) {
        for (int i = j + tid; i < n; i += B) {
            double t = G[(size_t)i * n + j];
            const double* Li = G + (size_t)i * n;
            const double* Lj = G + (size_t)j * n;
            for (int p = 0; p < j; ++p) t -= Li[p] * Lj[p];
            G[(size_t)i * n + j] = t; // t_i, not yet divided
        }
        __syncthreads();
        const double tj = G[(size_t)j * n + j];
        if (!(tj > 0.0) || !isfinite(tj)) {
            if (tid == 0) sh.bad = 1;
            __syncthreads();
            return;
        }
        const double ljj = sqrt(tj);
        if (tid == 0) G[(size_t)j * n + j] = ljj;
        for (int i = j + 1 + tid; i < n; i += B)
            G[(size_t)i * n + j] /= ljj;
        __syncthreads();
    }
    // Apply L^{-1}: vector j <- (vector j - sum_{p<j} L[j][p] vector' p)
    // / L[j][j]. Sequential over j (each vector needs the finished previous
    // ones), block-parallel over entries.
    for (int j = 0; j < n; ++j) {
        double* zj = z + (size_t)j * n;
        const double* Lj = G + (size_t)j * n;
        const double ljj = Lj[j];
        for (int i = tid; i < n; i += B) {
            double acc = zj[i];
            for (int p = 0; p < j; ++p) acc -= Lj[p] * z[(size_t)p * n + i];
            zj[i] = acc / ljj;
        }
        __syncthreads();
    }
}

__device__ inline void cholqr2_d(double* z, int n, double* G, ShCtx& sh) {
    cholqr_once_d(z, n, G, sh);
    if (!sh.bad) cholqr_once_d(z, n, G, sh);
}

// Inverse subspace iteration over the whole eigenvector block. w: the n
// Sturm eigenvalues. z: n vectors of n doubles (vector-major). gs: per-
// thread gtsv scratch, 3n doubles per column. G: n*n chol scratch.
__device__ inline void eigvecs_subspace_d(const double* d, const double* e,
                                          int n, const double* w,
                                          uint64_t bseed, double* z,
                                          double* gs, double* G,
                                          ShCtx& sh) {
    const int B = blockDim.x, tid = threadIdx.x;
    for (int k = tid; k < n; k += B) {
        uint64_t st = mix64_d(bseed + (uint64_t)k);
        double* col = z + (size_t)k * n;
        for (int i = 0; i < n; ++i) col[i] = lcg_uniform_d(st);
    }
    __syncthreads();
    cholqr2_d(z, n, G, sh);
    for (int it = 0; it < 4; ++it) {
        // All n shifted solves in parallel, in place (x may alias b).
        for (int k = tid; k < n; k += B) {
            const double lam = w[k];
            const double sig = lam + 1e-13 * (1.0 + fabs(lam));
            gtsv_shift_d(d, e, sig, n, z + (size_t)k * n,
                         z + (size_t)k * n, gs + (size_t)k * 3 * n);
        }
        __syncthreads();
        // Per-column clip to unit max-abs; deterministic refill when a
        // column blew up or vanished.
        for (int k = tid; k < n; k += B) {
            double* col = z + (size_t)k * n;
            double mx = 0.0;
            bool bad = false;
            for (int i = 0; i < n; ++i) {
                if (!isfinite(col[i])) { bad = true; break; }
                mx = fmax(mx, fabs(col[i]));
            }
            if (bad || mx == 0.0) {
                uint64_t st = mix64_d(bseed + (uint64_t)k +
                                      0x9e3779b9ull * (uint64_t)(it + 1));
                for (int i = 0; i < n; ++i) col[i] = lcg_uniform_d(st);
            } else {
                const double inv = 1.0 / mx;
                for (int i = 0; i < n; ++i) col[i] *= inv;
            }
        }
        __syncthreads();
        cholqr2_d(z, n, G, sh);
        if (sh.bad) { // chol breakdown: deterministic full-block refill
            __syncthreads();
            for (int k = tid; k < n; k += B) {
                uint64_t st = mix64_d(bseed + (uint64_t)k +
                                      0x517cc1b7ull * (uint64_t)(it + 2));
                double* col = z + (size_t)k * n;
                for (int i = 0; i < n; ++i) col[i] = lcg_uniform_d(st);
            }
            __syncthreads();
            cholqr2_d(z, n, G, sh);
        }
        // Convergence exit: per-column tridiagonal residual
        // ||T z_k - w_k z_k||_inf / (1 + |w_k|) after the orthonormalize;
        // the sweep stops when every column is below tol. Non-finite or
        // refilled columns carry a large residual, so the degenerate-
        // cluster path never exits early. Data-dependent but deterministic
        // (same input, same path), like the Sturm early exit. Inverse
        // iteration with a 1e-13 shift offset converges in 1-2 sweeps on
        // well-separated spectra, so the fixed 4-sweep count wastes work
        // on exactly the converged majority (the convergence-variance
        // hypothesis).
        double myres = 0.0;
        for (int k = tid; k < n; k += B) {
            const double* col = z + (size_t)k * n;
            const double lam = w[k];
            double r = 0.0;
            for (int i = 0; i < n; ++i) {
                const double ri =
                    (d[i] - lam) * col[i] +
                    ((i > 0) ? e[i - 1] * col[i - 1] : 0.0) +
                    ((i + 1 < n) ? e[i] * col[i + 1] : 0.0);
                r = fmax(r, fabs(ri));
            }
            myres = fmax(myres, r / (1.0 + fabs(lam)));
        }
        const double maxres = block_max(myres, sh);
        if (maxres <= 1e-13) break;
    }
    // Sign convention: largest-|entry| component positive (same as the
    // per-vector path).
    for (int k = tid; k < n; k += B) {
        double* col = z + (size_t)k * n;
        int imx = 0;
        double mx = 0.0;
        for (int i = 0; i < n; ++i)
            if (fabs(col[i]) > mx) { mx = fabs(col[i]); imx = i; }
        if (col[imx] < 0.0)
            for (int i = 0; i < n; ++i) col[i] = -col[i];
    }
    __syncthreads();
}

struct DevWork {
    dc* a;
    dc* vr;
    dc* cs;
    double *d, *e, *tau, *phase, *z, *ds;
};

__host__ __device__ inline int work_complex_count_d(int n) {
    return 2 * n * n + 2 * n;
}
__host__ __device__ inline int work_double_count_d(int n) {
    return n * n + 8 * n;
}

__device__ inline DevWork make_work_d(dc* cbuf, double* dbuf, int n) {
    DevWork W;
    dc* cp = cbuf;
    double* dp = dbuf;
    W.a = cp;  cp += (size_t)n * n;
    W.vr = cp; cp += (size_t)n * n;
    W.cs = cp;
    W.d = dp;     dp += n;
    W.e = dp;     dp += n;
    W.tau = dp;   dp += n;
    W.phase = dp; dp += n;
    W.z = dp;     dp += (size_t)n * n;
    W.ds = dp;
    return W;
}

__host__ __device__ inline int work_double_count_d2(int n) {
    return 5 * n * n + 8 * n; // z + 3n-per-column gtsv scratch + gram
}

// Same pipeline as eig_batch_kernel with the subspace eigenvector phase.
__global__ void eig2_batch_kernel(const dc* __restrict__ Abatch, int n,
                                  uint64_t seed, double* __restrict__ wout,
                                  dc* __restrict__ vout, dc* carena,
                                  double* darena) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x, B = blockDim.x;
    __shared__ ShCtx sh;
    DevWork W = make_work_d(carena + (size_t)b * work_complex_count_d(n),
                            darena + (size_t)b * work_double_count_d2(n), n);
    double* gs = W.ds + 4 * n;           // 3n per column
    double* G = gs + 3 * (size_t)n * n;  // n*n
    const dc* A = Abatch + (size_t)b * n * n;
    for (int idx = tid; idx < n * n; idx += B) W.a[idx] = A[idx];
    __syncthreads();
    tridiag_hermitian_d(W.a, n, W.d, W.e, W.vr, W.tau, W.phase, W.cs, sh);
    double* w = wout + (size_t)b * n;
    for (int k = tid; k < n; k += B) w[k] = find_eigen_d(W.d, W.e, n, k);
    __syncthreads();
    eigvecs_subspace_d(W.d, W.e, n, w, mix64_d(seed + (uint64_t)b), W.z, gs,
                       G, sh);
    back_transform_d(W.vr, W.tau, W.phase, W.z, n,
                     vout + (size_t)b * n * n);
}

// One block per matrix. Abatch: batch slabs of row-major n*n Hermitian
// (reinterpreted cdouble). wout: batch slabs of n. vout: batch slabs of
// col-major n*n. Matrix b solves at base seed mix64(seed + b), matching the
// CPU batch API. carena/darena: batch slabs of the work counts above.
__global__ void eig_batch_kernel(const dc* __restrict__ Abatch, int n,
                                 uint64_t seed, double* __restrict__ wout,
                                 dc* __restrict__ vout, dc* carena,
                                 double* darena) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x, B = blockDim.x;
    __shared__ ShCtx sh;
    DevWork W = make_work_d(carena + (size_t)b * work_complex_count_d(n),
                            darena + (size_t)b * work_double_count_d(n), n);
    const dc* A = Abatch + (size_t)b * n * n;
    for (int idx = tid; idx < n * n; idx += B) W.a[idx] = A[idx];
    __syncthreads();
    tridiag_hermitian_d(W.a, n, W.d, W.e, W.vr, W.tau, W.phase, W.cs, sh);
    double* w = wout + (size_t)b * n;
    for (int k = tid; k < n; k += B) w[k] = find_eigen_d(W.d, W.e, n, k);
    __syncthreads();
    const uint64_t bseed = mix64_d(seed + (uint64_t)b);
    for (int k = 0; k < n; ++k)
        eigvec_inverse_iter_d(W.d, W.e, n, k, w[k],
                              mix64_d(bseed + (uint64_t)k), W.z, W.ds, sh);
    back_transform_d(W.vr, W.tau, W.phase, W.z, n,
                     vout + (size_t)b * n * n);
}

inline void cuda_check(cudaError_t st, const char* what) {
    if (st != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(st));
}

// Device-resident batch. Upload once, solve many times (benchmarks time the
// kernel with inputs resident, KPM discipline).
struct Batch {
    int n = 0, batch = 0;
    dc* a = nullptr;       // batch*n*n
    double* w = nullptr;   // batch*n
    dc* v = nullptr;       // batch*n*n
    dc* carena = nullptr;  // batch*work_complex_count_d(n)
    double* darena = nullptr;
    double* darena2 = nullptr; // subspace variant, lazy (solve2)

    Batch(const cdouble* h_a, int n_, int batch_) : n(n_), batch(batch_) {
        if (n_ < 1 || batch_ < 1 || n_ > kMaxSharedN)
            throw std::invalid_argument(
                "beig gpu: need 1 <= n <= kMaxSharedN and batch >= 1");
        const size_t asz = (size_t)batch * n * n;
        cuda_check(cudaMalloc(&a, asz * sizeof(dc)), "cudaMalloc a");
        cuda_check(cudaMalloc(&w, (size_t)batch * n * sizeof(double)),
                   "cudaMalloc w");
        cuda_check(cudaMalloc(&v, asz * sizeof(dc)), "cudaMalloc v");
        cuda_check(cudaMalloc(&carena,
                              (size_t)batch * work_complex_count_d(n) *
                                  sizeof(dc)),
                   "cudaMalloc carena");
        cuda_check(cudaMalloc(&darena,
                              (size_t)batch * work_double_count_d(n) *
                                  sizeof(double)),
                   "cudaMalloc darena");
        upload(h_a);
    }
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;
    ~Batch() {
        cudaFree(a);
        cudaFree(w);
        cudaFree(v);
        cudaFree(carena);
        cudaFree(darena);
        cudaFree(darena2);
    }
    void upload(const cdouble* h_a) {
        cuda_check(cudaMemcpy(a, h_a, (size_t)batch * n * n * sizeof(dc),
                              cudaMemcpyHostToDevice),
                   "H2D a");
    }
    void solve(uint64_t seed) {
        eig_batch_kernel<<<batch, kThreads>>>(a, n, seed, w, v, carena,
                                              darena);
        cuda_check(cudaGetLastError(), "eig_batch_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "eig_batch_kernel");
    }
    // Subspace-iteration variant (eig2). Allocates the larger arena on
    // first use.
    void solve2(uint64_t seed) {
        if (!darena2)
            cuda_check(cudaMalloc(&darena2,
                                  (size_t)batch * work_double_count_d2(n) *
                                      sizeof(double)),
                       "cudaMalloc darena2");
        eig2_batch_kernel<<<batch, kThreads>>>(a, n, seed, w, v, carena,
                                               darena2);
        cuda_check(cudaGetLastError(), "eig2_batch_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "eig2_batch_kernel");
    }
    void download(std::vector<double>& h_w, std::vector<cdouble>& h_v) const {
        h_w.resize((size_t)batch * n);
        h_v.resize((size_t)batch * n * n);
        cuda_check(cudaMemcpy(h_w.data(), w,
                              (size_t)batch * n * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "D2H w");
        cuda_check(cudaMemcpy(h_v.data(), v,
                              (size_t)batch * n * n * sizeof(dc),
                              cudaMemcpyDeviceToHost),
                   "D2H v");
    }
};

// Convenience: full solve host-to-host. Matrix b at seed mix64(seed + b),
// matching solve_batch.
inline void solve_batch_gpu(const cdouble* h_a, int n, int batch,
                            uint64_t seed, std::vector<double>& h_w,
                            std::vector<cdouble>& h_v) {
    Batch B(h_a, n, batch);
    B.solve(seed);
    B.download(h_w, h_v);
}

// Subspace-iteration variant, same contract.
inline void solve_batch_gpu2(const cdouble* h_a, int n, int batch,
                             uint64_t seed, std::vector<double>& h_w,
                             std::vector<cdouble>& h_v) {
    Batch B(h_a, n, batch);
    B.solve2(seed);
    B.download(h_w, h_v);
}

} // namespace gpu
} // namespace beig
