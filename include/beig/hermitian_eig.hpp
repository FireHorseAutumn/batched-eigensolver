#pragma once

// hermitian_eig.hpp — batched dense complex-Hermitian eigensolver, FP64.
//
// Per matrix: Householder reduction of A (row-major n x n Hermitian) to real
// symmetric tridiagonal T, Sturm bisection for each eigenvalue (Gershgorin
// bounds, fixed 80 iterations — sp_core pattern), shifted inverse iteration
// on T for eigenvectors with full Gram-Schmidt reorthogonalization against
// all previously accepted vectors (n <= 128, so clusters are not special-
// cased), then back-transformation through the stored reflectors.
//
// Tridiagonalization convention: A = Q T_c Q^H with Q = H_0 H_1 ... H_{n-3}
// the product of complex reflectors, and T_c = D T D^H with D a diagonal
// unitary (phase[i]) making T real with non-negative off-diagonal.
// Eigenvectors of A are u = Q D z for eigenvectors z of T.

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace beig {

using cdouble = std::complex<double>;

inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Uniform in (-1/2, 1/2), advancing the LCG state (sp_core pattern).
inline double lcg_uniform(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(s >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

// Count of eigenvalues of the symmetric tridiagonal (d[0..n-1], e[0..n-2])
// strictly below lam, via the Sturm sequence of leading principal minors.
inline int sturm_count(const double* d, const double* e, int n, double lam) {
    double q = d[0] - lam;
    int cnt = (q < 0.0) ? 1 : 0;
    for (int i = 1; i < n; ++i) {
        double qq = q;
        if (fabs(qq) < 1e-290) qq = (qq < 0.0) ? -1e-290 : 1e-290;
        q = (d[i] - lam) - e[i-1] * e[i-1] / qq;
        if (q < 0.0) ++cnt;
    }
    return cnt;
}

// k-th smallest eigenvalue (0-based) by Sturm bisection from Gershgorin
// bounds; fixed 80 iterations -> deterministic, saturates at ULP resolution.
inline double find_eigen(const double* d, const double* e, int n, int k) {
    double lo = d[0], hi = d[0];
    for (int i = 0; i < n; ++i) {
        double r = 0.0;
        if (i > 0)     r += fabs(e[i-1]);
        if (i < n - 1) r += fabs(e[i]);
        lo = fmin(lo, d[i] - r);
        hi = fmax(hi, d[i] + r);
    }
    lo -= 1.0; hi += 1.0;
    for (int it = 0; it < 80; ++it) {
        double mid = 0.5 * (lo + hi);
        if (sturm_count(d, e, n, mid) >= k + 1) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}

// (T - sig*I) x = b for symmetric tridiagonal (d, e), Gaussian elimination
// with partial pivoting (LAPACK dgtsv recurrence; du2 collects the fill-in
// one past the superdiagonal on row swaps). Pivots are floored at 1e-290:
// inverse iteration drives T - sig*I exactly singular on degenerate
// eigenvalues, and a huge finite solve is harmless after normalization.
// s: 3n scratch. x may alias b.
inline void gtsv_shift(const double* d, const double* e, double sig, int n,
                       const double* b, double* x, double* s) {
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

// Householder reduction of Hermitian a (row-major n x n, overwritten) to real
// symmetric tridiagonal (d[0..n-1], e[0..n-2]). Row k of vr (columns k+1..n-1)
// receives the reflector vector v_k, tau[k] = 2/(v_k^H v_k) (0 for an empty
// column); H_k = I - tau[k] v_k v_k^H acting on rows/cols k+1..n-1.
// phase[i] = arg of D = diag(exp(i*phase[i])) with T_c = D T D^H.
// The trailing submatrix is re-symmetrized after each rank-2 update; the
// update is Hermitian in exact arithmetic but not bit-exactly in FP64.
// cs: 2n complex scratch.
inline void tridiag_hermitian(cdouble* a, int n, double* d, double* e,
                              cdouble* vr, double* tau, double* phase,
                              cdouble* cs) {
    cdouble* w = cs;
    cdouble* u = cs + n;
    for (int k = 0; k < n - 2; ++k) {
        const int m = n - k - 1;
        cdouble* v = vr + (size_t)k * n + k + 1;
        double nrm = 0.0;
        for (int i = 0; i < m; ++i) {
            v[i] = a[(size_t)(k + 1 + i) * n + k];
            nrm += std::norm(v[i]);
        }
        nrm = sqrt(nrm);
        if (nrm == 0.0) {
            tau[k] = 0.0;
            a[(size_t)(k + 1) * n + k] = 0.0;
            a[(size_t)k * n + k + 1] = 0.0;
            continue;
        }
        const cdouble x0 = v[0];
        const cdouble ph = (abs(x0) > 0.0) ? x0 / abs(x0) : cdouble(1.0, 0.0);
        const cdouble alpha = -ph * nrm; // H_k x = alpha*e_1, |alpha| = ||x||
        v[0] = x0 - alpha;
        // v^H v = 2*||x||*(||x|| + |x_0|); the |x_0| + ||x|| form avoids
        // cancellation in v_0.
        tau[k] = 1.0 / (nrm * (nrm + abs(x0)));
        for (int p = 0; p < m; ++p) {
            const cdouble* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            cdouble acc = 0.0;
            for (int q = 0; q < m; ++q) acc += srow[q] * v[q];
            w[p] = acc;
        }
        double s2 = 0.0; // v^H S v is real for Hermitian S; Im part is rounding
        for (int p = 0; p < m; ++p) s2 += (conj(v[p]) * w[p]).real();
        // S' = (I - tau v v^H) S (I - tau v v^H) = S - v u^H - u v^H
        const double g = 0.5 * tau[k] * tau[k] * s2;
        for (int p = 0; p < m; ++p) u[p] = tau[k] * w[p] - g * v[p];
        for (int p = 0; p < m; ++p) {
            cdouble* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            for (int q = 0; q < m; ++q)
                srow[q] -= v[p] * conj(u[q]) + u[p] * conj(v[q]);
        }
        for (int p = 0; p < m; ++p) {
            cdouble* srow = a + (size_t)(k + 1 + p) * n + k + 1;
            srow[p] = srow[p].real();
            for (int q = p + 1; q < m; ++q) {
                const cdouble avg =
                    0.5 * (srow[q] + conj(a[(size_t)(k + 1 + q) * n + k + 1 + p]));
                srow[q] = avg;
                a[(size_t)(k + 1 + q) * n + k + 1 + p] = conj(avg);
            }
        }
        a[(size_t)(k + 1) * n + k] = alpha;
        a[(size_t)k * n + k + 1] = conj(alpha);
        for (int i = k + 2; i < n; ++i) {
            a[(size_t)i * n + k] = 0.0;
            a[(size_t)k * n + i] = 0.0;
        }
    }
    for (int i = 0; i < n; ++i) d[i] = a[(size_t)i * n + i].real();
    for (int i = 0; i < n - 1; ++i)
        e[i] = abs(a[(size_t)(i + 1) * n + i]);
    if (n >= 2) tau[n - 2] = 0.0;
    tau[n - 1] = 0.0;
    phase[0] = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        const cdouble c = a[(size_t)(i + 1) * n + i];
        phase[i + 1] = phase[i] + atan2(c.imag(), c.real());
    }
}

// Modified Gram-Schmidt, two passes, against columns 0..k-1 of z (n*n
// col-major unit vectors), then 2-normalization of col. If col is depleted
// (lies in the span of the previous columns) it is refilled from the LCG.
inline void orthonormalize(double* z, int k, int n, double* col, uint64_t& st) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        for (int pass = 0; pass < 2; ++pass)
            for (int j = 0; j < k; ++j) {
                const double* zj = z + (size_t)j * n;
                double s = 0.0;
                for (int i = 0; i < n; ++i) s += zj[i] * col[i];
                for (int i = 0; i < n; ++i) col[i] -= s * zj[i];
            }
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += col[i] * col[i];
        if (ss > 1e-24) {
            ss = sqrt(ss);
            for (int i = 0; i < n; ++i) col[i] /= ss;
            return;
        }
        for (int i = 0; i < n; ++i) col[i] = lcg_uniform(st);
    }
}

// Eigenvector of (d, e) for eigenvalue lam, orthonormal against columns
// 0..k-1 of z (n*n col-major, already filled); result into column k.
// Shifted inverse iteration, fixed 4 sweeps, reorthogonalized every sweep.
// sig stays 1e-13 relative off lam: close enough that one solve amplifies the
// target direction by ~gap/1e-13, far enough that gtsv stays finite.
// The per-sweep max-abs scaling (not 2-norm) keeps the amplified solves
// overflow-free. s: 4n scratch. Deterministic given seed.
inline void eigvec_inverse_iter(const double* d, const double* e, int n, int k,
                                double lam, uint64_t seed, double* z,
                                double* s) {
    double* y = s + 3 * n;
    double* col = z + (size_t)k * n;
    const double sig = lam + 1e-13 * (1.0 + fabs(lam));
    uint64_t st = seed;
    for (int i = 0; i < n; ++i) col[i] = lcg_uniform(st);
    orthonormalize(z, k, n, col, st);
    for (int it = 0; it < 4; ++it) {
        gtsv_shift(d, e, sig, n, col, y, s);
        double mx = 0.0;
        bool bad = false;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(y[i])) { bad = true; break; }
            mx = fmax(mx, fabs(y[i]));
        }
        if (bad || mx == 0.0) {
            for (int i = 0; i < n; ++i) col[i] = lcg_uniform(st);
        } else {
            const double inv = 1.0 / mx;
            for (int i = 0; i < n; ++i) col[i] = y[i] * inv;
        }
        orthonormalize(z, k, n, col, st);
    }
    int imx = 0;
    double mx = 0.0;
    for (int i = 0; i < n; ++i)
        if (fabs(col[i]) > mx) { mx = fabs(col[i]); imx = i; }
    if (col[imx] < 0.0)
        for (int i = 0; i < n; ++i) col[i] = -col[i];
}

// out = Q D z for all columns of z (n*n col-major) into out (n*n col-major
// complex): apply D, then H_{n-3} .. H_0.
inline void back_transform(const cdouble* vr, const double* tau,
                           const double* phase, const double* z, int n,
                           cdouble* out) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out[(size_t)j * n + i] =
                z[(size_t)j * n + i] * cdouble(cos(phase[i]), sin(phase[i]));
    for (int k = n - 3; k >= 0; --k) {
        if (tau[k] == 0.0) continue;
        const int m = n - k - 1;
        const cdouble* v = vr + (size_t)k * n + k + 1;
        for (int j = 0; j < n; ++j) {
            cdouble* yj = out + (size_t)j * n + k + 1;
            cdouble s = 0.0;
            for (int i = 0; i < m; ++i) s += conj(v[i]) * yj[i];
            s *= tau[k];
            for (int i = 0; i < m; ++i) yj[i] -= s * v[i];
        }
    }
}

struct Work {
    cdouble* a;    // n*n, overwritten copy of the input
    cdouble* vr;   // n*n reflector storage
    cdouble* cs;   // 2n
    double* d;     // n
    double* e;     // n
    double* tau;   // n
    double* phase; // n
    double* z;     // n*n, real eigenvectors of T (col-major)
    double* ds;    // 4n
};

inline int work_complex_count(int n) { return 2 * n * n + 2 * n; }
inline int work_double_count(int n) { return n * n + 8 * n; }

inline Work make_work(cdouble* cbuf, double* dbuf, int n) {
    Work W;
    cdouble* cp = cbuf;
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

// Full eigendecomposition of Hermitian A (row-major n*n, unchanged on
// return): w[0..n-1] ascending eigenvalues, V (n*n col-major) column j =
// eigenvector j with V^H V = I.
inline void hermitian_eig(const cdouble* A, int n, double* w, cdouble* V,
                          uint64_t seed, Work& W) {
    for (size_t i = 0; i < (size_t)n * n; ++i) W.a[i] = A[i];
    tridiag_hermitian(W.a, n, W.d, W.e, W.vr, W.tau, W.phase, W.cs);
    for (int k = 0; k < n; ++k) {
        w[k] = find_eigen(W.d, W.e, n, k);
        eigvec_inverse_iter(W.d, W.e, n, k, w[k], mix64(seed + (uint64_t)k),
                            W.z, W.ds);
    }
    back_transform(W.vr, W.tau, W.phase, W.z, n, V);
}

struct EigResult {
    int n = 0;
    std::vector<double> w;     // ascending
    std::vector<cdouble> v;    // n*n col-major, column j = eigenvector j
};

inline EigResult hermitian_eig(const cdouble* A, int n, uint64_t seed) {
    EigResult r;
    r.n = n;
    r.w.resize(n);
    r.v.resize((size_t)n * n);
    std::vector<cdouble> cbuf(work_complex_count(n));
    std::vector<double> dbuf(work_double_count(n));
    Work W = make_work(cbuf.data(), dbuf.data(), n);
    hermitian_eig(A, n, r.w.data(), r.v.data(), seed, W);
    return r;
}

struct BatchItem {
    int n;
    const cdouble* a; // row-major n*n Hermitian, unchanged
    double* w;        // out, n ascending eigenvalues
    cdouble* v;       // out, n*n col-major eigenvectors
};

// Item b is solved with seed mix64(seed + b), so batched output is bitwise
// reproducible matrix-by-matrix from the base seed.
inline void solve_batch(BatchItem* items, int batch, uint64_t seed) {
    for (int b = 0; b < batch; ++b) {
        BatchItem& it = items[b];
        std::vector<cdouble> cbuf(work_complex_count(it.n));
        std::vector<double> dbuf(work_double_count(it.n));
        Work W = make_work(cbuf.data(), dbuf.data(), it.n);
        hermitian_eig(it.a, it.n, it.w, it.v, mix64(seed + (uint64_t)b), W);
    }
}

} // namespace beig
