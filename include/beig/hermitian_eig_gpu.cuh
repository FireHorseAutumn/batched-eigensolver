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
// sequential — Sturm runs one eigenvalue per thread, gtsv on thread 0; the
// O(n^2)/O(n^3) inner loops (norms, S*v, rank-2 update, re-symmetrization,
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

// Same fixed-80-iteration bisection as the CPU find_eigen. Runs one
// eigenvalue per thread.
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
// The re-symmetrization is race-free: thread p owns row p's upper triangle
// and column p's lower triangle exclusively.
__device__ inline void tridiag_hermitian_d(dc* a, int n, double* d, double* e,
                                           dc* vr, double* tau, double* phase,
                                           dc* cs, ShCtx& sh) {
    dc* w = cs;
    dc* u = cs + n;
    const int B = blockDim.x, tid = threadIdx.x;
    for (int k = 0; k < n - 2; ++k) {
        const int m = n - k - 1;
        dc* v = vr + (size_t)k * n + k + 1;
        double mine = 0.0;
        for (int i = tid; i < m; i += B) {
            const dc val = a[(size_t)(k + 1 + i) * n + k];
            v[i] = val;
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
            const dc x0 = v[0];
            const double ax0 = dabs(x0);
            const dc ph = (ax0 > 0.0) ? dscale(x0, 1.0 / ax0) : dc{1.0, 0.0};
            const dc alpha = dscale(ph, -nrm);
            sh.are = alpha.re;
            sh.aim = alpha.im;
            tau[k] = 1.0 / (nrm * (nrm + ax0));
            v[0] = dsub(x0, alpha);
        }
        __syncthreads();
        const double tauk = tau[k];
        for (int p = tid; p < m; p += B) {
            const dc* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            dc acc = {0.0, 0.0};
            for (int q = 0; q < m; ++q) acc = dadd(acc, dmul(srow[q], v[q]));
            w[p] = acc;
        }
        __syncthreads();
        double mine2 = 0.0;
        for (int p = tid; p < m; p += B)
            mine2 += dmul(dconj(v[p]), w[p]).re;
        const double s2 = block_sum(mine2, sh);
        const double g = 0.5 * tauk * tauk * s2;
        for (int p = tid; p < m; p += B)
            u[p] = dsub(dscale(w[p], tauk), dscale(v[p], g));
        __syncthreads();
        for (int p = tid; p < m; p += B) {
            dc* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            for (int q = 0; q < m; ++q)
                srow[q] = dsub(srow[q],
                               dadd(dmul(v[p], dconj(u[q])),
                                    dmul(u[p], dconj(v[q]))));
        }
        __syncthreads();
        for (int p = tid; p < m; p += B) {
            dc* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            srow[p] = {srow[p].re, 0.0};
            for (int q = p + 1; q < m; ++q) {
                dc* other = &a[(size_t)(k + 1 + q) * n + k + 1 + p];
                const dc avg = dscale(dadd(srow[q], dconj(*other)), 0.5);
                srow[q] = avg;
                *other = dconj(avg);
            }
        }
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
// then reflectors applied one column per thread.
__device__ inline void back_transform_d(const dc* vr, const double* tau,
                                        const double* phase, const double* z,
                                        int n, dc* out) {
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
        const dc* v = vr + (size_t)k * n + k + 1;
        for (int j = tid; j < n; j += B) {
            dc* yj = out + (size_t)j * n + k + 1;
            dc s = {0.0, 0.0};
            for (int i = 0; i < m; ++i)
                s = dadd(s, dmul(dconj(v[i]), yj[i]));
            s = dscale(s, tau[k]);
            for (int i = 0; i < m; ++i)
                yj[i] = dsub(yj[i], dmul(s, v[i]));
        }
        __syncthreads();
    }
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
    if (tid < n) w[tid] = find_eigen_d(W.d, W.e, n, tid);
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

    Batch(const cdouble* h_a, int n_, int batch_) : n(n_), batch(batch_) {
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

} // namespace gpu
} // namespace beig
