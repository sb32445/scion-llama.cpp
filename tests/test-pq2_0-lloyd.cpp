// Unit test for the Lloyd-Max refinement of the PQ2_0 group scale.
//
// This is the deployment quantizer of the ternary-serve MoE correction recipe
// (opt-in via GGML_PQ2_0_LLOYD=1); Prism's absmax rule stays the default, so
// this is the only place the Lloyd path is exercised in-tree.  It checks the
// properties the recipe relies on:
//
//   1. codes stay in {0,1,2} -- the +2 state the container allows is never
//      emitted, so the tensors remain ternary in {-1,0,+1};
//   2. a block whose magnitudes all exceed half the scale converges to the
//      TWN fixed point a = mean(|w|) (within fp16 storage tolerance);
//   3. a block with a clear large/small split puts the scale on the mean of
//      the large magnitudes (the small ones collapse to the zero state);
//   4. an all-zero block is finite (d = 0, all codes the zero state);
//   5. the result is byte-deterministic.

#include "../ggml/src/ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...)                          \
    do {                                          \
        if (!(cond)) {                            \
            printf("  FAIL: ");                   \
            printf(__VA_ARGS__);                  \
            printf("\n");                         \
            ++n_fail;                             \
        }                                         \
    } while (0)

static int code_of(const block_pq2_0 & b, int j) {
    return (b.qs[j / 4] >> (2 * (j % 4))) & 3;
}

static float dequant(const block_pq2_0 & b, int j) {
    return (float) (code_of(b, j) - 1) * ggml_fp16_to_fp32(b.d);
}

static void check_codes_ternary(const block_pq2_0 * b, int nb, const char * what) {
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < QK_PQ2_0; ++j) {
            CHECK(code_of(b[i], j) <= 2, "%s: block %d code[%d] = %d (not ternary)", what, i, j, code_of(b[i], j));
        }
    }
}

static void test_random_codes_and_determinism() {
    printf("random blocks: codes + determinism\n");

    std::mt19937 gen(12345);
    std::normal_distribution<float> dis(0.0f, 1.0f);

    const int nb = 16;
    std::vector<float> x(QK_PQ2_0 * nb);
    for (auto & v : x) {
        v = dis(gen);
    }

    std::vector<block_pq2_0> a(nb);
    std::vector<block_pq2_0> b(nb);
    quantize_row_pq2_0_lloyd_ref(x.data(), a.data(), (int64_t) x.size());
    quantize_row_pq2_0_lloyd_ref(x.data(), b.data(), (int64_t) x.size());

    check_codes_ternary(a.data(), nb, "random");
    CHECK(memcmp(a.data(), b.data(), sizeof(block_pq2_0) * nb) == 0, "not deterministic");

    // the quantizer must actually use the zero state (else it is not ternary)
    int zeros = 0;
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < QK_PQ2_0; ++j) {
            zeros += code_of(a[i], j) == 1;
        }
    }
    CHECK(zeros > 0, "no zero-state codes in %d random blocks", nb);
}

static void test_fixed_point_all_above_threshold() {
    printf("fixed point: all |w| above half the scale\n");

    std::mt19937 gen(999);
    std::uniform_real_distribution<float> mag(0.9f, 1.1f);

    float x[QK_PQ2_0];
    double sum = 0.0;
    for (int j = 0; j < QK_PQ2_0; ++j) {
        const float w = mag(gen) * ((j % 2) ? -1.0f : 1.0f);
        x[j] = w;
        sum += fabsf(w);
    }
    const double mean = sum / QK_PQ2_0;

    block_pq2_0 b;
    quantize_row_pq2_0_lloyd_ref(x, &b, QK_PQ2_0);

    check_codes_ternary(&b, 1, "fixed-point");

    const double d = ggml_fp16_to_fp32(b.d);
    CHECK(fabs(d - mean) <= 1e-3 * mean, "scale %.6f != mean %.6f (rel %.2e)", d, mean, fabs(d - mean) / mean);

    // every weight is above d/2, so no code may be the zero state
    for (int j = 0; j < QK_PQ2_0; ++j) {
        CHECK(code_of(b, j) != 1, "weight %.4f collapsed to zero (d = %.6f)", x[j], d);
    }
}

static void test_fixed_point_split() {
    printf("fixed point: large/small split\n");

    float x[QK_PQ2_0];
    double large_sum = 0.0;
    const int n_large = 8;
    for (int j = 0; j < QK_PQ2_0; ++j) {
        if (j < n_large) {
            const float w = 2.0f + 0.01f * (j % 5);
            x[j] = w;
            large_sum += w;
        } else {
            x[j] = 1e-3f * ((j % 7) + 1);
        }
    }
    const double large_mean = large_sum / n_large;

    block_pq2_0 b;
    quantize_row_pq2_0_lloyd_ref(x, &b, QK_PQ2_0);

    check_codes_ternary(&b, 1, "split");

    const double d = ggml_fp16_to_fp32(b.d);
    CHECK(fabs(d - large_mean) <= 2e-3 * large_mean, "scale %.6f != mean(large) %.6f", d, large_mean);

    // the large values survive, the small ones collapse
    for (int j = 0; j < n_large; ++j) {
        CHECK(code_of(b, j) != 1, "large weight %.4f collapsed (d = %.6f)", x[j], d);
        CHECK(fabs(dequant(b, j) - x[j]) < 0.05, "large weight %.4f -> %.4f", x[j], dequant(b, j));
    }
    for (int j = n_large; j < QK_PQ2_0; ++j) {
        CHECK(code_of(b, j) == 1, "small weight %.4f did not collapse (d = %.6f)", x[j], d);
    }
}

static void test_zero_block() {
    printf("zero block\n");

    float x[QK_PQ2_0] = {};
    block_pq2_0 b;
    quantize_row_pq2_0_lloyd_ref(x, &b, QK_PQ2_0);

    CHECK(b.d == 0, "zero block stored d = %u", (unsigned) b.d);
    for (int j = 0; j < QK_PQ2_0; ++j) {
        CHECK(code_of(b, j) == 1, "zero block code[%d] = %d", j, code_of(b, j));
        CHECK(dequant(b, j) == 0.0f, "zero block dequant[%d] = %f", j, dequant(b, j));
    }
}

int main() {
    test_random_codes_and_determinism();
    test_fixed_point_all_above_threshold();
    test_fixed_point_split();
    test_zero_block();

    if (n_fail == 0) {
        printf("all pq2_0 Lloyd checks passed\n");
        return 0;
    }
    printf("%d pq2_0 Lloyd checks FAILED\n", n_fail);
    return 1;
}
