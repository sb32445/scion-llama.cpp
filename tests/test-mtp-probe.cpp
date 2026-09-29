// mtp-probe: dump per-position post-norm hidden states + greedy tokens from a
// target model over prepared token windows (Scion drafter lane, Phase E).
//
// Our frozen-body MTP drafter consumes the target's post-norm final hidden
// (the same tensor the LM head sees; qwen35moe exposes it as `t_embd` via the
// standard embeddings output) plus the embedding of the next token.  Training
// and acceptance for the *released* model therefore need those hidden states
// from the release itself -- the 4-layer prefix's geometry does not transfer.
//
// This tool runs fixed token windows through a model with embeddings enabled,
// requesting outputs on every row, and writes per window:
//   <out>/win_XX_h.bin       fp32  [seq, n_embd]  post-norm hidden per position
//   <out>/win_XX_argmax.bin  int32 [seq]          greedy token per position
// (row i predicts token i+1; the head's draft at position t is compared
// against argmax at row t+1 -- same alignment as moe/mtp_eval.py).
//
// usage: test-mtp-probe <model.gguf> <tokens.bin> <n_windows> <seq> <outdir>
//                      [n_gpu_layers=999]
//
// tokens.bin: int32 little-endian, n_windows * seq ids (build with the target
// tokenizer, e.g. the HF tokenizer the scion windows() uses).

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void fail(const std::string & msg) {
    fprintf(stderr, "mtp-probe: %s\n", msg.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <model.gguf> <tokens.bin> <n_windows> <seq> <outdir> "
                "[n_gpu_layers=999]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string tok_path   = argv[2];
    const int         n_windows  = atoi(argv[3]);
    const int         seq        = atoi(argv[4]);
    const std::string outdir     = argv[5];
    const int         n_gpu_layers = argc > 6 ? atoi(argv[6]) : 999;

    std::vector<int32_t> ids((size_t) n_windows * seq);
    FILE * f = fopen(tok_path.c_str(), "rb");
    if (!f) {
        fail("cannot open " + tok_path);
    }
    if (fread(ids.data(), sizeof(int32_t), ids.size(), f) != ids.size()) {
        fail("short read from " + tok_path);
    }
    fclose(f);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = n_gpu_layers;
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fail("failed to load model: " + model_path);
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = seq + 8;
    cparams.n_batch    = seq + 8;
    cparams.n_ubatch   = seq + 8;
    cparams.n_seq_max  = 1;
    cparams.no_perf    = true;
    cparams.embeddings = true;   // extract embeddings together with logits
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fail("failed to create context");
    }

    printf("model %s: n_embd %d, n_vocab %d, windows %d x %d\n",
           model_path.c_str(), n_embd, n_vocab, n_windows, seq);
    fflush(stdout);

    std::vector<float>   h((size_t) seq * n_embd);
    std::vector<int32_t> am(seq);

    for (int w = 0; w < n_windows; ++w) {
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(seq, 0, 1);
        for (int i = 0; i < seq; ++i) {
            batch.token[i]      = (llama_token) ids[(size_t) w * seq + i];
            batch.pos[i]        = i;
            batch.n_seq_id[i]   = 1;
            batch.seq_id[i][0]  = 0;
            batch.logits[i]     = true;
        }
        batch.n_tokens = seq;

        if (llama_decode(ctx, batch) != 0) {
            fail("llama_decode failed on window " + std::to_string(w));
        }
        for (int i = 0; i < seq; ++i) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (!lg) {
                fail("no logits at row " + std::to_string(i));
            }
            int best = 0;
            float bv = lg[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (lg[v] > bv) {
                    bv = lg[v];
                    best = v;
                }
            }
            am[i] = best;

            const float * hh = llama_get_embeddings_ith(ctx, i);
            if (!hh) {
                fail("no embeddings at row " + std::to_string(i) +
                     " -- context created without embeddings?");
            }
            memcpy(h.data() + (size_t) i * n_embd, hh, sizeof(float) * n_embd);
        }
        llama_batch_free(batch);

        char path[4096];
        snprintf(path, sizeof(path), "%s/win_%02d_h.bin", outdir.c_str(), w);
        FILE * fh = fopen(path, "wb");
        if (!fh) {
            fail("cannot write " + std::string(path));
        }
        fwrite(h.data(), sizeof(float), h.size(), fh);
        fclose(fh);

        snprintf(path, sizeof(path), "%s/win_%02d_argmax.bin", outdir.c_str(), w);
        FILE * fa = fopen(path, "wb");
        if (!fa) {
            fail("cannot write " + std::string(path));
        }
        fwrite(am.data(), sizeof(int32_t), am.size(), fa);
        fclose(fa);

        printf("window %d/%d done\n", w + 1, n_windows);
        fflush(stdout);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
