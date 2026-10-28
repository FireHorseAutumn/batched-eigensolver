// GPU port gate: one-block-per-matrix solver vs the CPU reference, which is
// itself Eigen-oracle-gated in test_eig. Per matrix: eigenvalue agreement
// |w_gpu - w_cpu| <= 1e-12 * max(1, ||H||_F) (both sides sit ~1e-14 from
// the oracle), plus absolute checks on the GPU output alone: eigenpair
// residual <= 1e-12 * max(1, ||H||_F), orthonormality <= 1e-11. GPU block
// reductions are tree-ordered, so bitwise equality with the CPU is NOT
// expected; run-to-run bitwise determinism of the kernel is checked
// separately (fixed geometry, fixed reduction shape).

#include "beig/hermitian_eig_gpu.cuh"
#include "ensembles.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using beig::cdouble;
int failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
    std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name, detail.c_str());
    if (!ok) ++failures;
}

double frobenius(const cdouble* A, int n) {
    double s = 0.0;
    for (size_t i = 0; i < (size_t)n * n; ++i)
        s += A[i].real() * A[i].real() + A[i].imag() * A[i].imag();
    return sqrt(s);
}

struct Worst {
    double eval = 0.0, res = 0.0, orth = 0.0;
};

// Fold one GPU-solved matrix into wst: eval agreement vs the CPU reference,
// residual and orthonormality of the GPU output against the input matrix.
void check_one(const cdouble* A, int n, const double* w_gpu,
               const cdouble* v_gpu, const double* w_cpu, Worst& wst) {
    const double scale = std::max(1.0, frobenius(A, n));
    for (int i = 0; i < n; ++i)
        wst.eval =
            std::max(wst.eval, std::abs(w_gpu[i] - w_cpu[i]) / scale);
    for (int i = 0; i < n; ++i) {
        const cdouble* v = v_gpu + (size_t)i * n;
        double acc = 0.0;
        for (int p = 0; p < n; ++p) {
            cdouble hv = 0.0;
            for (int q = 0; q < n; ++q) hv += A[(size_t)p * n + q] * v[q];
            acc += std::norm(hv - w_gpu[i] * v[p]);
        }
        wst.res = std::max(wst.res, sqrt(acc) / scale);
    }
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
            const cdouble* vi = v_gpu + (size_t)i * n;
            const cdouble* vj = v_gpu + (size_t)j * n;
            cdouble acc = 0.0;
            for (int p = 0; p < n; ++p) acc += conj(vi[p]) * vj[p];
            wst.orth =
                std::max(wst.orth, std::abs(acc - (i == j ? 1.0 : 0.0)));
        }
}

// Note: the GPU paths store eigenvector i as the contiguous slab
// v_gpu + i*n (vector-major).

constexpr double TOL_EVAL = 1e-12; // times max(1, ||H||_F), vs CPU reference
constexpr double TOL_RES = 1e-12;  // times max(1, ||H||_F)
constexpr double TOL_ORTH = 1e-11; // absolute

void report(const char* name, const Worst& wst) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "eval-vs-cpu %.3e (tol %.0e*||H||_F), res %.3e "
                  "(tol %.0e*||H||_F), orth %.3e (tol %.0e)",
                  wst.eval, TOL_EVAL, wst.res, TOL_RES, wst.orth, TOL_ORTH);
    check(wst.eval < TOL_EVAL && wst.res < TOL_RES && wst.orth < TOL_ORTH,
          name, buf);
}

// Solve a batch on one GPU path (0 = per-vector, 1 = subspace) and fold
// every matrix into wst.
void run_batch(const std::vector<cdouble>& Abatch, int n, int batch,
               uint64_t seed, Worst& wst, std::vector<double>& w_gpu,
               std::vector<cdouble>& v_gpu, int path = 0) {
    if (path == 0)
        beig::gpu::solve_batch_gpu(Abatch.data(), n, batch, seed, w_gpu,
                                   v_gpu);
    else
        beig::gpu::solve_batch_gpu2(Abatch.data(), n, batch, seed, w_gpu,
                                    v_gpu);
    for (int b = 0; b < batch; ++b) {
        const beig::EigResult r =
            beig::hermitian_eig(Abatch.data() + (size_t)b * n * n, n,
                                beig::mix64(seed + (uint64_t)b));
        check_one(Abatch.data() + (size_t)b * n * n, n,
                  w_gpu.data() + (size_t)b * n,
                  v_gpu.data() + (size_t)b * n * n, r.w.data(), wst);
    }
}

std::vector<cdouble> make_batch(int n, int batch, uint64_t seed,
                                int ensemble) {
    std::vector<cdouble> out;
    for (int b = 0; b < batch; ++b) {
        std::vector<cdouble> A;
        const uint64_t s = beig::mix64(seed + (uint64_t)b);
        if (ensemble == 0)
            A = beigtest::random_hermitian(n, s);
        else if (ensemble == 1)
            A = beigtest::clustered_hermitian(n, s, 0.0);
        else
            A = beigtest::clustered_hermitian(n, s, 1e-8);
        out.insert(out.end(), A.begin(), A.end());
    }
    return out;
}

} // namespace

int main() {
    const uint64_t base = 0xB3162026ULL;

    for (int n : {32, 64, 100, 128, 144}) {
        const std::vector<cdouble> Ab =
            make_batch(n, 4, base + 0x1000 + 16 * n, 0);
        for (int path : {0, 1}) {
            Worst wst;
            std::vector<double> w_gpu;
            std::vector<cdouble> v_gpu;
            run_batch(Ab, n, 4, base + 0x1000 + 16 * n, wst, w_gpu, v_gpu,
                      path);
            char name[80];
            std::snprintf(name, sizeof name, "GPU%s GUE n=%d (batch 4)",
                          path ? " subspace" : "", n);
            report(name, wst);
            if (n == 64) {
                std::vector<double> w2;
                std::vector<cdouble> v2;
                if (path == 0)
                    beig::gpu::solve_batch_gpu(Ab.data(), n, 4,
                                               base + 0x1000 + 16 * n, w2,
                                               v2);
                else
                    beig::gpu::solve_batch_gpu2(Ab.data(), n, 4,
                                                base + 0x1000 + 16 * n, w2,
                                                v2);
                const bool same =
                    std::memcmp(w_gpu.data(), w2.data(),
                                w_gpu.size() * sizeof(double)) == 0 &&
                    std::memcmp(v_gpu.data(), v2.data(),
                                v_gpu.size() * sizeof(cdouble)) == 0;
                check(same,
                      path ? "GPU subspace run-to-run bitwise determinism "
                             "(n=64)"
                           : "GPU run-to-run bitwise determinism (n=64)",
                      same ? "w and V bit-identical across reruns"
                           : "MISMATCH across reruns");
            }
        }
    }

    for (int n : {64, 128}) {
        for (int ens : {1, 2}) {
            const std::vector<cdouble> Ab =
                make_batch(n, 4, base + 0x2000 + 16 * n + ens, ens);
            for (int path : {0, 1}) {
                Worst wst;
                std::vector<double> w_gpu;
                std::vector<cdouble> v_gpu;
                run_batch(Ab, n, 4, base + 0x2000 + 16 * n + ens, wst, w_gpu,
                          v_gpu, path);
                char name[96];
                std::snprintf(name, sizeof name,
                              "GPU%s clustered n=%d jitter=%s (batch 4)",
                              path ? " subspace" : "", n,
                              ens == 1 ? "0" : "1e-8");
                report(name, wst);
            }
        }
    }

    const int nks = 7;
    for (int W : {16, 32, 64}) {
        const int n = 2 * W;
        std::vector<cdouble> Ab;
        for (int j = 0; j < nks; ++j) {
            const double k = -3.14159265358979323846 +
                             (j + 0.5) * 2.0 * 3.14159265358979323846 / nks;
            std::vector<cdouble> A = beigtest::zigzag_ribbon(W, k);
            Ab.insert(Ab.end(), A.begin(), A.end());
        }
        for (int path : {0, 1}) {
            Worst wst;
            std::vector<double> w_gpu;
            std::vector<cdouble> v_gpu;
            run_batch(Ab, n, nks, base + 0x3000 + 64 * W, wst, w_gpu, v_gpu,
                      path);
            char name[96];
            std::snprintf(name, sizeof name,
                          "GPU%s zigzag ribbon W=%d n=%d (batch %d)",
                          path ? " subspace" : "", W, n, nks);
            report(name, wst);
        }
    }

    std::printf("%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
