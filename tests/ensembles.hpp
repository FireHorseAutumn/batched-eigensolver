#pragma once

// ensembles.hpp — seeded test ensembles for the oracle harness.
// All generators are deterministic in (seed, size); no <random>, same
// splitmix64 + Box-Muller everywhere so results are platform-independent.

#include "beig/hermitian_eig.hpp"

#include <cmath>
#include <vector>

namespace beigtest {

using beig::cdouble;

inline uint64_t sm64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline double uniform01(uint64_t& s) {
    return (double)(sm64(s) >> 11) * (1.0 / 9007199254740992.0);
}

// Standard normal via Box-Muller, one sample per call.
inline double gauss(uint64_t& s) {
    const double two_pi = 6.28318530717958647692;
    double u1 = uniform01(s);
    while (u1 < 1e-300) u1 = uniform01(s);
    return sqrt(-2.0 * log(u1)) * cos(two_pi * uniform01(s));
}

// Force bit-exact Hermiticity: lower triangle := conj(upper), diagonal real.
inline void symmetrize(std::vector<cdouble>& a, int n) {
    for (int i = 0; i < n; ++i) {
        a[(size_t)i * n + i] = a[(size_t)i * n + i].real();
        for (int j = i + 1; j < n; ++j)
            a[(size_t)j * n + i] = conj(a[(size_t)i * n + j]);
    }
}

// A = (B + B^H)/2 with B a complex Gaussian matrix (Ginibre): a GUE draw.
inline std::vector<cdouble> random_hermitian(int n, uint64_t seed) {
    uint64_t s = seed;
    std::vector<cdouble> b((size_t)n * n);
    for (auto& x : b) x = cdouble(gauss(s), gauss(s));
    std::vector<cdouble> a((size_t)n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a[(size_t)i * n + j] =
                0.5 * (b[(size_t)i * n + j] + conj(b[(size_t)j * n + i]));
    symmetrize(a, n);
    return a;
}

// A = U D U^H with U a random unitary (twice-modified Gram-Schmidt on a
// Gaussian matrix) and D integer-valued on {0..7}: every eigenvalue sits in
// an exactly degenerate cluster of multiplicity ~n/8. jitter > 0 adds
// jitter*N(0,1) to each level, turning the clusters near-degenerate instead.
inline std::vector<cdouble> clustered_hermitian(int n, uint64_t seed,
                                                double jitter) {
    uint64_t s = seed;
    std::vector<cdouble> u((size_t)n * n); // col-major
    for (auto& x : u) x = cdouble(gauss(s), gauss(s));
    for (int j = 0; j < n; ++j) {
        cdouble* uj = u.data() + (size_t)j * n;
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < j; ++i) {
                const cdouble* ui = u.data() + (size_t)i * n;
                cdouble r = 0.0;
                for (int p = 0; p < n; ++p) r += conj(ui[p]) * uj[p];
                for (int p = 0; p < n; ++p) uj[p] -= r * ui[p];
            }
        double ss = 0.0;
        for (int p = 0; p < n; ++p) ss += std::norm(uj[p]);
        ss = sqrt(ss);
        for (int p = 0; p < n; ++p) uj[p] /= ss;
    }
    std::vector<double> d(n);
    for (int i = 0; i < n; ++i)
        d[i] = floor(8.0 * uniform01(s)) + jitter * gauss(s);
    std::vector<cdouble> a((size_t)n * n, 0.0);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
            cdouble acc = 0.0;
            for (int i = 0; i < n; ++i)
                acc += d[i] * u[(size_t)p + (size_t)i * n] *
                       conj(u[(size_t)q + (size_t)i * n]);
            a[(size_t)p * n + q] = acc;
        }
    symmetrize(a, n);
    return a;
}

// Zigzag graphene nanoribbon, W zigzag chains across the width,
// nearest-neighbor hopping t (real), lattice period 1 along the ribbon.
// Basis: A_0..A_{W-1}, B_0..B_{W-1}; n = 2W. Per unit cell: A_j--B_j via
// displacements {0, -1}, giving t(1 + e^{-ik}); A_j--B_{j+1} for
// j = 0..W-2, intra-cell t. At k = pi the Bloch bond vanishes and the
// spectrum is exactly {0 (twice, the zigzag edge modes), ±|t| (W-1 each)}.
inline std::vector<cdouble> zigzag_ribbon(int W, double k, double t = -1.0) {
    const int n = 2 * W;
    std::vector<cdouble> H((size_t)n * n, 0.0);
    const cdouble bloch = t * cdouble(1.0 + cos(k), -sin(k));
    for (int j = 0; j < W; ++j) {
        const int A = j, B = W + j;
        H[(size_t)A * n + B] = bloch;
        H[(size_t)B * n + A] = conj(bloch);
        if (j + 1 < W) {
            H[(size_t)A * n + B + 1] = t;
            H[(size_t)(B + 1) * n + A] = t;
        }
    }
    return H;
}

} // namespace beigtest
