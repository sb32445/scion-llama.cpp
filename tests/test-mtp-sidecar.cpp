// test-mtp-sidecar: evaluate the Scion MTP sidecar head on dumped probe
// windows and report greedy acceptance + per-position head-eval timing.
//
// The head consumes the target's post-norm final hidden (the tensor the LM
// head sees; dumped by test-mtp-probe into win_XX_h.bin) plus the raw GGUF
// token-embedding row of the NEXT token, and predicts the target's greedy
// token at that position (win_XX_argmax.bin).  The final projection reuses the
// target's own output_norm + output tensors, read straight from the release
// GGUF -- exactly the tensors the runtime sidecar path will reuse in VRAM.
//
// Graph (per position p in [0, seq-2)):
//   h   = hidden[p]                              (post-norm, fp32)
//   e   = token_embd[ids[p+1]]                   (raw stored row)
//   x   = concat(rms(h), rms(e))                 [2H]
//   z   = fc2 @ gelu_erf(fc1 @ x)                [H]
//   lg  = output @ (rms(z) * output_norm)        [vocab]
//   hit = argmax(lg) == argmax_target[p+1]
//
// usage: test-mtp-sidecar <sidecar.gguf> <model.gguf> <probe_dir> <win_first> <win_last>
//                          [seq=512] [cpu]
//
// cpu selects the CPU backend; default prefers the first GPU backend.
// The sidecar is expected to carry mtp.fc1.weight / mtp.fc2.weight (F16) and
// mtp.hidden_size / mtp.width metadata (see moe/mtp_sidecar_export.py).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void fail(const std::string & msg) {
    fprintf(stderr, "mtp-sidecar: %s\n", msg.c_str());
    exit(1);
}

struct gguf_tensor_data {
    enum ggml_type type;
    int64_t        ne[4];
    size_t         nbytes;
    std::vector<uint8_t> bytes;
};

// read one tensor's raw bytes from a GGUF without materializing the whole file
static gguf_tensor_data read_gguf_tensor(const std::string & path, const std::string & name) {
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    struct gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        fail("cannot open gguf " + path);
    }
    const int64_t tid = gguf_find_tensor(g, name.c_str());
    if (tid < 0) {
        fail("gguf " + path + " has no tensor " + name);
    }

    gguf_tensor_data out;
    out.type   = gguf_get_tensor_type(g, tid);
    out.nbytes = gguf_get_tensor_size(g, tid);
    const int64_t * ne = gguf_get_tensor_ne(g, tid);
    for (int i = 0; i < 4; ++i) {
        out.ne[i] = ne[i];
    }
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
    gguf_free(g);

    out.bytes.resize(out.nbytes);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fail("cannot reopen " + path);
    }
    if (fseek(f, (long) off, SEEK_SET) != 0 ||
        fread(out.bytes.data(), 1, out.nbytes, f) != out.nbytes) {
        fail("short read of " + name + " from " + path);
    }
    fclose(f);
    return out;
}

static bool gguf_meta_f32(const std::string & path, const char * key, float * out) {
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        return false;
    }
    const int64_t kid = gguf_find_key(g, key);
    bool ok = kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_FLOAT32;
    if (ok) {
        *out = gguf_get_val_f32(g, kid);
    }
    gguf_free(g);
    return ok;
}

static bool gguf_meta_u32(const std::string & path, const char * key, uint32_t * out) {
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        return false;
    }
    const int64_t kid = gguf_find_key(g, key);
    bool ok = kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_UINT32;
    if (ok) {
        *out = gguf_get_val_u32(g, kid);
    }
    gguf_free(g);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <sidecar.gguf> <model.gguf> <probe_dir> <win_first> <win_last> "
                "[seq=512] [cpu]\n", argv[0]);
        return 1;
    }
    const std::string sidecar_path = argv[1];
    const std::string model_path   = argv[2];
    const std::string probe_dir    = argv[3];
    const int win_first = atoi(argv[4]);
    const int win_last  = atoi(argv[5]);
    const int seq       = argc > 6 ? atoi(argv[6]) : 512;
    const bool use_cpu  = argc > 7 && strcmp(argv[7], "cpu") == 0;
    const bool mode_gen = argc > 8 && strcmp(argv[8], "gen") == 0;

    if (win_last <= win_first || seq < 4) {
        fail("bad window range or seq");
    }

    // sidecar metadata
    uint32_t md_hidden = 0, md_width = 0;
    if (!gguf_meta_u32(sidecar_path, "mtp.hidden_size", &md_hidden) ||
        !gguf_meta_u32(sidecar_path, "mtp.width", &md_width)) {
        fail("sidecar metadata missing mtp.hidden_size / mtp.width");
    }
    float claimed_fw = -1.0f, claimed_wt = -1.0f;
    gguf_meta_f32(sidecar_path, "mtp.acceptance.fineweb",  &claimed_fw);
    gguf_meta_f32(sidecar_path, "mtp.acceptance.wikitext", &claimed_wt);

    // sidecar + model tensors (raw bytes)
    const gguf_tensor_data fc1 = read_gguf_tensor(sidecar_path, "mtp.fc1.weight");
    const gguf_tensor_data fc2 = read_gguf_tensor(sidecar_path, "mtp.fc2.weight");
    const gguf_tensor_data out = read_gguf_tensor(model_path, "output.weight");
    const gguf_tensor_data tok = read_gguf_tensor(model_path, "token_embd.weight");
    const gguf_tensor_data nrm = read_gguf_tensor(model_path, "output_norm.weight");

    const int64_t H     = out.ne[0];
    const int64_t vocab = out.ne[1];
    const int64_t W     = fc1.ne[1];

    if (fc1.ne[0] != 2*H || fc2.ne[0] != W || fc2.ne[1] != H ||
        (int64_t) md_hidden != H || (int64_t) md_width != W ||
        tok.ne[0] != H || tok.ne[1] != vocab || nrm.ne[0] != H) {
        fail("shape mismatch between sidecar, model and expectations");
    }

    const int C = seq - 2; // positions that have both a target and a next token

    llama_backend_init();
    ggml_backend_load_all();

    ggml_backend_dev_t dev = use_cpu
        ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
        : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fail("failed to init backend");
    }

    printf("sidecar %s: H %d, W %d (fp16); model vocab %d\n",
           sidecar_path.c_str(), (int) H, (int) W, (int) vocab);
    printf("backend %s\n", ggml_backend_dev_name(dev));
    fflush(stdout);

    // context: weights + one fixed-shape graph (C positions per window)
    struct ggml_init_params ip = {
        /*.mem_size   =*/ 64u*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fail("ggml_init failed");
    }

    struct ggml_tensor * t_fc1 = ggml_new_tensor_2d(ctx, fc1.type, fc1.ne[0], fc1.ne[1]);
    struct ggml_tensor * t_fc2 = ggml_new_tensor_2d(ctx, fc2.type, fc2.ne[0], fc2.ne[1]);
    struct ggml_tensor * t_out = ggml_new_tensor_2d(ctx, out.type, out.ne[0], out.ne[1]);
    struct ggml_tensor * t_tok = ggml_new_tensor_2d(ctx, tok.type, tok.ne[0], tok.ne[1]);
    struct ggml_tensor * t_nrm = ggml_new_tensor_1d(ctx, nrm.type, nrm.ne[0]);

    // head graph builder, used for both the batched (C positions) and the
    // deployment-shape (1 position) graph
    struct head_graph {
        ggml_tensor * h   = nullptr;
        ggml_tensor * ids = nullptr;
        ggml_tensor * am  = nullptr;
        ggml_cgraph * gf  = nullptr;
    };
    auto build_head = [&](int n) {
        head_graph g;
        g.h   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n);
        g.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_tensor * e  = ggml_get_rows(ctx, t_tok, g.ids);        // [H, n]
        ggml_tensor * hn = ggml_rms_norm(ctx, g.h, 1e-6f);          // [H, n]
        ggml_tensor * en = ggml_rms_norm(ctx, e, 1e-6f);            // [H, n]
        ggml_tensor * x  = ggml_concat(ctx, hn, en, 0);             // [2H, n]
        ggml_tensor * z1 = ggml_mul_mat(ctx, t_fc1, x);             // [W, n]
        ggml_tensor * z2 = ggml_gelu_erf(ctx, z1);
        ggml_tensor * z  = ggml_mul_mat(ctx, t_fc2, z2);            // [H, n]
        ggml_tensor * zn = ggml_rms_norm(ctx, z, 1e-6f);            // [H, n]
        ggml_tensor * zs = ggml_mul(ctx, zn, t_nrm);                // broadcast [H]
        ggml_tensor * lg = ggml_mul_mat(ctx, t_out, zs);            // [vocab, n]
        g.am = ggml_argmax(ctx, lg);                                // [n] I32
        return g;
    };

    head_graph g_batch = build_head(C);
    head_graph g_one   = build_head(1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fail("failed to allocate tensors on backend");
    }

    ggml_backend_tensor_set(t_fc1, fc1.bytes.data(), 0, fc1.bytes.size());
    ggml_backend_tensor_set(t_fc2, fc2.bytes.data(), 0, fc2.bytes.size());
    ggml_backend_tensor_set(t_out, out.bytes.data(), 0, out.bytes.size());
    ggml_backend_tensor_set(t_tok, tok.bytes.data(), 0, tok.bytes.size());
    ggml_backend_tensor_set(t_nrm, nrm.bytes.data(), 0, nrm.bytes.size());

    g_batch.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(g_batch.gf, g_batch.am);
    g_one.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(g_one.gf, g_one.am);

    ggml_tensor * t_h   = g_batch.h;
    ggml_tensor * t_ids = g_batch.ids;
    ggml_tensor * t_am  = g_batch.am;
    ggml_cgraph * gf    = g_batch.gf;

    // ---------------------------------------------------------------- gen ----
    // Greedy generation with k=1 sidecar drafting, using the exact runtime
    // inputs: the target's post-norm hidden from the nextn output + the raw
    // embedding row of the just-sampled token.  Loop invariant per step:
    //   KV holds rows 0..pos-1, `h_prev` is the hidden at row pos-1, `cur` is
    //   the next token to decode (not yet in the KV).  One target forward per
    //   step; accepted drafts yield 2 emitted tokens (cur + draft).
    if (mode_gen) {
        const int n_steps = 256;

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 999;
        llama_model * lmodel = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!lmodel) { fail("failed to load model for gen mode"); }
        const llama_vocab * voc = llama_model_get_vocab(lmodel);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = seq + n_steps + 8;
        cparams.n_batch   = seq + n_steps + 8;
        cparams.n_ubatch  = seq + n_steps + 8;
        cparams.n_seq_max = 1;
        cparams.no_perf   = true;
        llama_context * lctx = llama_init_from_model(lmodel, cparams);
        if (!lctx) { fail("failed to create model context for gen mode"); }
        llama_set_embeddings_nextn(lctx, true, /*masked*/ false);

        std::vector<int32_t> gids(seq);
        std::vector<float> h_prev(H), h_cur(H);
        int32_t id_in = 0;

        auto argmax_f32 = [&](const float * v) {
            int32_t best = 0;
            float   bv   = v[0];
            for (int64_t i = 1; i < vocab; ++i) {
                if (v[i] > bv) { bv = v[i]; best = (int32_t) i; }
            }
            return best;
        };
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto ms_since = [](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };

        int64_t n_draft = 0, n_accept = 0, n_tok = 0, n_fwd = 0, n_copy = 0;
        double  draft_ms = 0.0;
        double  fwd_ms   = 0.0;

        for (int w = win_first; w < win_last; ++w) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/tokens.bin", probe_dir.c_str());
            FILE * f = fopen(path, "rb");
            if (!f) { fail("cannot open tokens.bin"); }
            if (fseek(f, (long) ((size_t) w * seq * sizeof(int32_t)), SEEK_SET) != 0 ||
                fread(gids.data(), sizeof(int32_t), gids.size(), f) != gids.size()) {
                fail("short tokens read");
            }
            fclose(f);

            llama_memory_clear(llama_get_memory(lctx), true);

            llama_batch batch = llama_batch_init(seq + n_steps + 8, 0, 1);
            for (int i = 0; i < seq; ++i) {
                batch.token[i]     = gids[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (i == seq - 1);
            }
            batch.n_tokens = seq;
            if (llama_decode(lctx, batch) != 0) { fail("prefill decode failed"); }

            const float * hp = llama_get_embeddings_nextn_ith(lctx, seq - 1);
            if (!hp) { fail("no nextn hidden after prefill"); }
            memcpy(h_prev.data(), hp, (size_t) H * sizeof(float));

            llama_token cur = argmax_f32(llama_get_logits_ith(lctx, seq - 1));
            int pos = seq;
            int64_t w_acc = 0, w_fwd = 0;

            for (int it = 0; it < n_steps && cur != llama_vocab_eos(voc); ++it) {
                // draft: head(h at row pos-1, embedding of the token to decode)
                ggml_backend_tensor_set(g_one.h, h_prev.data(), 0, (size_t) H * sizeof(float));
                id_in = (int32_t) cur;
                ggml_backend_tensor_set(g_one.ids, &id_in, 0, sizeof(int32_t));
                const auto d0 = now();
                if (ggml_backend_graph_compute(backend, g_one.gf) != GGML_STATUS_SUCCESS) {
                    fail("draft graph compute failed");
                }
                ggml_backend_synchronize(backend);
                int32_t draft = -1;
                ggml_backend_tensor_get(g_one.am, &draft, 0, sizeof(int32_t));
                draft_ms += ms_since(d0);
                n_draft++;

                // verify: decode the pending token
                batch.n_tokens     = 1;
                batch.token[0]     = cur;
                batch.pos[0]       = pos;
                batch.n_seq_id[0]  = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0]    = true;
                const auto f0 = now();
                if (llama_decode(lctx, batch) != 0) { fail("decode failed"); }
                n_fwd++; w_fwd++;
                pos++;

                const llama_token z2 = argmax_f32(llama_get_logits_ith(lctx, 0));
                const float * hc = llama_get_embeddings_nextn_ith(lctx, 0);
                if (!hc) { fail("no nextn hidden after decode"); }
                fwd_ms += ms_since(f0);
                memcpy(h_cur.data(), hc, (size_t) H * sizeof(float));

                if (z2 == cur) { n_copy++; }
                if (z2 == (llama_token) draft) {
                    n_accept++; w_acc++;
                    n_tok += 2;
                    cur = (llama_token) draft;
                } else {
                    n_tok += 1;
                    cur = z2;
                }
                memcpy(h_prev.data(), h_cur.data(), (size_t) H * sizeof(float));
            }

            printf("window %d: %lld forwards, %lld/%lld accepted, %lld tokens\n",
                   w, (long long) w_fwd, (long long) w_acc, (long long) w_fwd,
                   (long long) (w_fwd + w_acc));
            fflush(stdout);
            llama_batch_free(batch);
        }

        printf("---\n");
        printf("gen: %lld forwards, %lld drafts, %lld accepted (%.4f)\n",
               (long long) n_fwd, (long long) n_draft, (long long) n_accept,
               n_draft ? (double) n_accept / n_draft : 0.0);
        printf("gen: copy-last control acceptance %.4f (%lld/%lld)\n",
               n_fwd ? (double) n_copy / n_fwd : 0.0, (long long) n_copy, (long long) n_fwd);
        printf("gen: %lld tokens generated = %.3f tokens/forward\n",
               (long long) n_tok, n_fwd ? (double) n_tok / n_fwd : 0.0);
        printf("gen: draft %.3f ms/eval\n", n_draft ? draft_ms / n_draft : 0.0);
        printf("gen: target forward %.3f ms/forward\n", n_fwd ? fwd_ms / n_fwd : 0.0);
        printf("gen: speedup estimate %.3fx (tokens/forward / (1 + draft/forward))\n",
               (n_fwd && fwd_ms > 0.0)
                   ? ((double) n_tok / n_fwd) / (1.0 + (draft_ms / n_draft) / (fwd_ms / n_fwd))
                   : 0.0);
        llama_free(lctx);
        llama_model_free(lmodel);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        llama_backend_free();
        return 0;
    }

    // per-window dumps
    std::vector<float>   h((size_t) seq * H);
    std::vector<int32_t> ids(seq);
    std::vector<int32_t> tgt(seq);
    std::vector<int32_t> am(C);

    std::vector<float>   toks(C);
    std::vector<float>   hh((size_t) C * H);

    int64_t n_hit = 0, n_tot = 0;
    std::vector<double> ms_per_window;

    for (int w = win_first; w < win_last; ++w) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/h/win_%02d_h.bin", probe_dir.c_str(), w);
        FILE * f = fopen(path, "rb");
        if (!f) { fail("cannot open " + std::string(path)); }
        if (fread(h.data(), sizeof(float), h.size(), f) != h.size()) { fail("short h read"); }
        fclose(f);

        snprintf(path, sizeof(path), "%s/h/win_%02d_argmax.bin", probe_dir.c_str(), w);
        f = fopen(path, "rb");
        if (!f) { fail("cannot open " + std::string(path)); }
        if (fread(tgt.data(), sizeof(int32_t), tgt.size(), f) != tgt.size()) { fail("short argmax read"); }
        fclose(f);

        snprintf(path, sizeof(path), "%s/tokens.bin", probe_dir.c_str());
        f = fopen(path, "rb");
        if (!f) { fail("cannot open " + std::string(path)); }
        if (fseek(f, (long) ((size_t) w * seq * sizeof(int32_t)), SEEK_SET) != 0 ||
            fread(ids.data(), sizeof(int32_t), ids.size(), f) != ids.size()) {
            fail("short tokens read");
        }
        fclose(f);

        // h rows 0..C-1; next token = ids[1..C]; target = argmax[1..C]
        memcpy(hh.data(), h.data(), (size_t) C * H * sizeof(float));
        std::vector<int32_t> nxt(C);
        for (int i = 0; i < C; ++i) {
            nxt[i] = ids[i + 1];
        }
        ggml_backend_tensor_set(t_h,   hh.data(),  0, (size_t) C * H * sizeof(float));
        ggml_backend_tensor_set(t_ids, nxt.data(), 0, (size_t) C * sizeof(int32_t));

        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }
        ggml_backend_synchronize(backend);
        const auto t1 = std::chrono::steady_clock::now();

        ggml_backend_tensor_get(t_am, am.data(), 0, (size_t) C * sizeof(int32_t));

        int64_t hit = 0;
        for (int i = 0; i < C; ++i) {
            if (am[i] == tgt[i + 1]) {
                hit++;
            }
        }
        n_hit += hit;
        n_tot += C;
        ms_per_window.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        printf("window %d: acc %.4f (%lld/%d)  %.1f ms\n",
               w, (double) hit / C, (long long) hit, C, ms_per_window.back());
        fflush(stdout);
    }

    std::vector<double> sorted = ms_per_window;
    std::sort(sorted.begin(), sorted.end());
    const double ms_med = sorted[sorted.size()/2];

    // deployment-shape timing: one position per eval (the batched number
    // understates the per-draft cost: a single row cannot amortize the output
    // weight read over 510 columns)
    {
        std::vector<float>   h1((size_t) H);
        std::vector<int32_t> id1(1);
        int32_t              one = 0;
        memcpy(h1.data(), h.data(), (size_t) H * sizeof(float));
        ggml_backend_tensor_set(g_one.h, h1.data(), 0, (size_t) H * sizeof(float));

        const int reps = 200;
        for (int i = 0; i < 5; ++i) {
            id1[0] = ids[1 + (i % (seq - 1))];
            ggml_backend_tensor_set(g_one.ids, id1.data(), 0, sizeof(int32_t));
            ggml_backend_graph_compute(backend, g_one.gf);
        }
        ggml_backend_synchronize(backend);

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            id1[0] = ids[1 + (i % (seq - 1))];
            ggml_backend_tensor_set(g_one.ids, id1.data(), 0, sizeof(int32_t));
            if (ggml_backend_graph_compute(backend, g_one.gf) != GGML_STATUS_SUCCESS) {
                fail("single-position graph compute failed");
            }
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(g_one.am, &one, 0, sizeof(int32_t));
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms_one = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        printf("single-position head eval: %.3f ms/eval (compute+sync+argmax, %d reps, argmax %d)\n",
               ms_one, reps, one);
    }

    printf("---\n");
    printf("windows %d..%d: acceptance %.4f (%lld/%lld)\n",
           win_first, win_last - 1, (double) n_hit / n_tot, (long long) n_hit, (long long) n_tot);
    if (claimed_fw >= 0.0f || claimed_wt >= 0.0f) {
        printf("sidecar metadata claims: fineweb %.4f wikitext %.4f\n", claimed_fw, claimed_wt);
    }
    printf("head eval %.1f ms / %d-position window = %.3f ms/position (median)\n",
           ms_med, C, ms_med / C);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    llama_backend_free();
    return 0;
}
