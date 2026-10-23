// Oracle harness: beig vs Eigen SelfAdjointEigenSolver.
// Per matrix: max |lambda_i - lambda_i^Eigen|, max eigenpair residual
// ||H v_i - lambda_i v_i||_2, and max |v_i^H v_j - delta_ij|.
// Tolerances (stated, printed with every line):
//   eigenvalues : 1e-12 * max(1, ||H||_F)
//   residual    : 1e-12 * max(1, ||H||_F)
//   orthonormal : 1e-11 absolute
// Measured worst cases land ~1e-14 relative; the tolerances are ~100x above.

#include "beig/hermitian_eig.hpp"
#include "ensembles.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using beig::cdouble;
int failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
    std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name, detail.c_str());
    if (!ok) ++failures;
}

struct Worst {
    double eval = 0.0, res = 0.0, orth = 0.0;
};

// Solve one matrix, compare against Eigen, fold into wst.
void run_one(const std::vector<cdouble>& A, int n, uint64_t seed, Worst& wst) {
    const beig::EigResult r = beig::hermitian_eig(A.data(), n, seed);

    Eigen::MatrixXcd H(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) H(i, j) = A[(size_t)i * n + j];
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    if (es.info() != Eigen::Success) {
        check(false, "Eigen oracle", "SelfAdjointEigenSolver failed");
        return;
    }
    const double fro = H.norm();
    const double scale = std::max(1.0, fro);

    for (int i = 0; i < n; ++i)
        wst.eval = std::max(wst.eval,
                            std::abs(r.w[i] - es.eigenvalues()[i]) / scale);

    for (int i = 0; i < n; ++i) {
        const cdouble* v = r.v.data() + (size_t)i * n;
        double acc = 0.0;
        for (int p = 0; p < n; ++p) {
            cdouble hv = 0.0;
            for (int q = 0; q < n; ++q)
                hv += A[(size_t)p * n + q] * v[q];
            acc += std::norm(hv - r.w[i] * v[p]);
        }
        wst.res = std::max(wst.res, sqrt(acc) / scale);
    }

    for (int i = 0; i < n; ++i) {
        const cdouble* vi = r.v.data() + (size_t)i * n;
        for (int j = i; j < n; ++j) {
            const cdouble* vj = r.v.data() + (size_t)j * n;
            cdouble acc = 0.0;
            for (int p = 0; p < n; ++p) acc += conj(vi[p]) * vj[p];
            wst.orth = std::max(wst.orth, std::abs(acc - (i == j ? 1.0 : 0.0)));
        }
    }
}

constexpr double TOL_EVAL = 1e-12; // times max(1, ||H||_F)
constexpr double TOL_RES  = 1e-12; // times max(1, ||H||_F)
constexpr double TOL_ORTH = 1e-11; // absolute

void report(const char* name, const Worst& wst) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "eval %.3e (tol %.0e*||H||_F), res %.3e (tol %.0e*||H||_F), "
                  "orth %.3e (tol %.0e)",
                  wst.eval, TOL_EVAL, wst.res, TOL_RES, wst.orth, TOL_ORTH);
    check(wst.eval < TOL_EVAL && wst.res < TOL_RES && wst.orth < TOL_ORTH,
          name, buf);
}

} // namespace

int main() {
    const uint64_t base = 0xB3162026ULL;

    // GUE ensemble, sizes across the target range, 4 draws each.
    for (int n : {32, 64, 100, 128}) {
        Worst wst;
        for (int r = 0; r < 4; ++r) {
            const uint64_t seed = beig::mix64(base + 0x100 + 16 * n + r);
            run_one(beigtest::random_hermitian(n, seed), n, seed, wst);
        }
        char name[64];
        std::snprintf(name, sizeof name, "GUE n=%d (4 draws)", n);
        report(name, wst);
    }

    // Exactly degenerate clusters (integer levels 0..7), and near-degenerate
    // clusters (jitter 1e-8 per level).
    for (int n : {64, 128}) {
        for (double jitter : {0.0, 1e-8}) {
            Worst wst;
            for (int r = 0; r < 4; ++r) {
                const uint64_t seed =
                    beig::mix64(base + 0x200 + 16 * n + r);
                run_one(beigtest::clustered_hermitian(n, seed, jitter), n,
                        seed, wst);
            }
            char name[64];
            std::snprintf(name, sizeof name, "clustered n=%d jitter=%g (4 draws)",
                          n, jitter);
            report(name, wst);
        }
    }

    // Zigzag graphene ribbons (n = 2W), 7 generic k-points each.
    const int nks = 7;
    for (int W : {16, 32, 64}) {
        Worst wst;
        for (int j = 0; j < nks; ++j) {
            const double k = -3.14159265358979323846 +
                             (j + 0.5) * 2.0 * 3.14159265358979323846 / nks;
            const uint64_t seed = beig::mix64(base + 0x300 + 64 * W + j);
            run_one(beigtest::zigzag_ribbon(W, k), 2 * W, seed, wst);
        }
        char name[64];
        std::snprintf(name, sizeof name, "zigzag ribbon W=%d n=%d (%d k-points)",
                      W, 2 * W, nks);
        report(name, wst);
    }

    std::printf("%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
