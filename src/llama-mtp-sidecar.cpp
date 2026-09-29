#include "llama-mtp-sidecar.h"

#include "ggml-alloc.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <vector>

// raw tensor bytes from a GGUF without materializing the whole file
static bool read_gguf_tensor(const char * path, const char * name, std::vector<uint8_t> & bytes, enum ggml_type & type, int64_t ne[4], std::string & err) {
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    struct gguf_context * g = gguf_init_from_file(path, params);
    if (!g) {
        err = std::string("cannot open gguf ") + path;
        return false;
    }
    const int64_t tid = gguf_find_tensor(g, name);
    if (tid < 0) {
        gguf_free(g);
        err = std::string("gguf ") + path + " has no tensor " + name;
        return false;
    }
    type = gguf_get_tensor_type(g, tid);
    const size_t nbytes = gguf_get_tensor_size(g, tid);
    const int64_t * tne = gguf_get_tensor_ne(g, tid);
    for (int i = 0; i < 4; ++i) {
        ne[i] = tne[i];
    }
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
    gguf_free(g);

    bytes.resize(nbytes);
    FILE * f = fopen(path, "rb");
    if (!f) {
        err = std::string("cannot reopen ") + path;
        return false;
    }
    if (fseek(f, (long) off, SEEK_SET) != 0 || fread(bytes.data(), 1, nbytes, f) != nbytes) {
        fclose(f);
        err = std::string("short read of ") + name;
        return false;
    }
    fclose(f);
    return true;
}

static bool gguf_meta_u32(const char * path, const char * key, uint32_t & out) {
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(path, params);
    if (!g) {
        return false;
    }
    const int64_t kid = gguf_find_key(g, key);
    const bool ok = kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_UINT32;
    if (ok) {
        out = gguf_get_val_u32(g, kid);
    }
    gguf_free(g);
    return ok;
}

struct llama_mtp_sidecar::impl {
    const llama_model & model;

    int64_t hidden = 0;
    int64_t vocab  = 0;
    int64_t width  = 0;

    struct ggml_context * ctx   = nullptr;  // graph tensors
    struct ggml_context * ctx_w = nullptr;  // sidecar weights (external buffer)
    ggml_backend_buffer_t wbuf  = nullptr;
    ggml_backend_sched_t  sched = nullptr;
    struct ggml_cgraph  * gf = nullptr;

    struct ggml_tensor * fc1  = nullptr;
    struct ggml_tensor * fc2  = nullptr;
    struct ggml_tensor * h_in = nullptr;
    struct ggml_tensor * ids  = nullptr;
    struct ggml_tensor * am   = nullptr;
    struct ggml_tensor * lg   = nullptr;

    explicit impl(const llama_model & model) : model(model) {}
};

llama_mtp_sidecar::llama_mtp_sidecar(const llama_model & model) : pimpl(new impl(model)) {}

llama_mtp_sidecar::~llama_mtp_sidecar() {
    if (pimpl->sched) {
        ggml_backend_sched_free(pimpl->sched);
    }
    if (pimpl->wbuf) {
        ggml_backend_buffer_free(pimpl->wbuf);
    }
    if (pimpl->ctx) {
        ggml_free(pimpl->ctx);
    }
    if (pimpl->ctx_w) {
        ggml_free(pimpl->ctx_w);
    }
}

bool llama_mtp_sidecar::loaded() const {
    return pimpl->gf != nullptr;
}

bool llama_mtp_sidecar::load(const char * path, const std::vector<ggml_backend_t> & backends, std::string & err) {
    auto & p = *pimpl;

    if (backends.empty()) {
        err = "no backends available";
        return false;
    }
    if (p.gf) {
        err = "sidecar already loaded";
        return false;
    }
    if (!p.model.output_norm || !p.model.output || !p.model.tok_embd) {
        err = "target model lacks output_norm/output/tok_embd";
        return false;
    }

    uint32_t md_hidden = 0, md_width = 0;
    if (!gguf_meta_u32(path, "mtp.hidden_size", md_hidden) || !gguf_meta_u32(path, "mtp.width", md_width)) {
        err = "sidecar metadata missing mtp.hidden_size / mtp.width";
        return false;
    }

    std::vector<uint8_t> fc1_bytes, fc2_bytes;
    enum ggml_type fc1_type = GGML_TYPE_COUNT, fc2_type = GGML_TYPE_COUNT;
    int64_t fc1_ne[4] = {1,1,1,1}, fc2_ne[4] = {1,1,1,1};
    if (!read_gguf_tensor(path, "mtp.fc1.weight", fc1_bytes, fc1_type, fc1_ne, err) ||
        !read_gguf_tensor(path, "mtp.fc2.weight", fc2_bytes, fc2_type, fc2_ne, err)) {
        return false;
    }

    p.hidden = p.model.output->ne[0];
    p.vocab  = p.model.output->ne[1];
    p.width  = fc1_ne[1];

    if (fc1_ne[0] != 2*p.hidden || fc2_ne[0] != p.width || fc2_ne[1] != p.hidden ||
        (int64_t) md_hidden != p.hidden || (int64_t) md_width != p.width ||
        p.model.tok_embd->ne[0] != p.hidden || p.model.tok_embd->ne[1] != p.vocab ||
        p.model.output_norm->ne[0] != p.hidden) {
        err = "shape mismatch between sidecar, model and expectations";
        return false;
    }

    struct ggml_init_params ip = {
        /*.mem_size   =*/ 8u*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    p.ctx_w = ggml_init(ip);
    p.ctx   = ggml_init(ip);
    if (!p.ctx_w || !p.ctx) {
        err = "ggml_init failed";
        return false;
    }

    // the sidecar weights live in their own buffer, like model weights, so the
    // per-graph scheduler never reallocates (and clobbers) them
    ggml_backend_t w_backend = backends.back();
    {
        ggml_backend_buffer_type_t w_buft = ggml_backend_buffer_get_type(p.model.output->buffer);
        for (ggml_backend_t b : backends) {
            if (ggml_backend_supports_buft(b, w_buft)) {
                w_backend = b;
                break;
            }
        }
    }
    p.fc1 = ggml_new_tensor_2d(p.ctx_w, fc1_type, fc1_ne[0], fc1_ne[1]);
    p.fc2 = ggml_new_tensor_2d(p.ctx_w, fc2_type, fc2_ne[0], fc2_ne[1]);
    p.wbuf = ggml_backend_alloc_ctx_tensors(p.ctx_w, w_backend);
    if (!p.wbuf) {
        err = "failed to allocate sidecar weights";
        return false;
    }
    ggml_backend_tensor_set(p.fc1, fc1_bytes.data(), 0, fc1_bytes.size());
    ggml_backend_tensor_set(p.fc2, fc2_bytes.data(), 0, fc2_bytes.size());

    p.h_in = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, p.hidden, 1);
    p.ids  = ggml_new_tensor_1d(p.ctx, GGML_TYPE_I32, 1);
    ggml_set_input(p.h_in);
    ggml_set_input(p.ids);

    struct ggml_tensor * e  = ggml_get_rows(p.ctx, p.model.tok_embd, p.ids);
    struct ggml_tensor * hn = ggml_rms_norm(p.ctx, p.h_in, 1e-6f);
    struct ggml_tensor * en = ggml_rms_norm(p.ctx, e, 1e-6f);
    struct ggml_tensor * x  = ggml_concat(p.ctx, hn, en, 0);
    struct ggml_tensor * z1 = ggml_mul_mat(p.ctx, p.fc1, x);
    struct ggml_tensor * z2 = ggml_gelu_erf(p.ctx, z1);
    struct ggml_tensor * z  = ggml_mul_mat(p.ctx, p.fc2, z2);
    struct ggml_tensor * zn = ggml_rms_norm(p.ctx, z, 1e-6f);
    struct ggml_tensor * zs = ggml_mul(p.ctx, zn, p.model.output_norm);
    p.lg = ggml_mul_mat(p.ctx, p.model.output, zs);
    p.am = ggml_argmax(p.ctx, p.lg);

    p.gf = ggml_new_graph_custom(p.ctx, 128, false);
    ggml_build_forward_expand(p.gf, p.am);

    std::vector<ggml_backend_t> backs = backends;
    p.sched = ggml_backend_sched_new(backs.data(), nullptr,
                                     (int) backs.size(), 128, /*parallel=*/false, /*op_offload=*/true);
    if (!p.sched) {
        err = "failed to create sidecar scheduler";
        return false;
    }
    if (!ggml_backend_sched_alloc_graph(p.sched, p.gf)) {
        err = "failed to allocate sidecar graph";
        return false;
    }

    if (getenv("LLAMA_MTP_SIDECAR_DEBUG")) {
        float tmp[8] = {0};
        ggml_backend_tensor_get(p.fc1, tmp, 0, sizeof(tmp));
        LLAMA_LOG_INFO("%s: dbg fc1[0..7] = %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                       __func__, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
    }

    LLAMA_LOG_INFO("%s: sidecar '%s' loaded (hidden %d, width %d, vocab %d)\n",
                   __func__, path, (int) p.hidden, (int) p.width, (int) p.vocab);
    return true;
}

bool llama_mtp_sidecar::draft(const float * h, int32_t token, int32_t & token_out, std::string & err) {
    auto & p = *pimpl;
    if (!p.gf) {
        err = "sidecar not loaded";
        return false;
    }

    // the graph is static (one position, fixed tensors): the single allocation
    // from load() is reused, so the scheduler cannot re-plan away the weights
    // or the input copies

    ggml_backend_tensor_set(p.h_in, h, 0, (size_t) p.hidden * sizeof(float));
    ggml_backend_tensor_set(p.ids,  &token, 0, sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(p.sched, p.gf) != GGML_STATUS_SUCCESS) {
        err = "sidecar graph compute failed";
        return false;
    }
    ggml_backend_sched_synchronize(p.sched);

    if (getenv("LLAMA_MTP_SIDECAR_DEBUG")) {
        static int n_dbg = 0;
        if (n_dbg++ < 2) {
            float tmp[8] = {0};
            float hh[4]  = {0};
            std::vector<float> lgs(p.vocab, 0.0f);
            ggml_backend_tensor_get(p.fc1, tmp, 0, sizeof(tmp));
            ggml_backend_tensor_get(p.h_in, hh, 0, sizeof(hh));
            ggml_backend_tensor_get(p.lg, lgs.data(), 0, lgs.size() * sizeof(float));
            int amax = 0;
            for (int64_t i = 1; i < p.vocab; ++i) { if (lgs[i] > lgs[amax]) amax = (int) i; }
            LLAMA_LOG_INFO("%s: dbg draft %d fc1[0..3] = %.5f %.5f %.5f %.5f | h[0..3] = %.5f %.5f %.5f %.5f | lg[0..3] = %.3f %.3f %.3f %.3f | max %.3f @ %d\n",
                           __func__, n_dbg, tmp[0], tmp[1], tmp[2], tmp[3], hh[0], hh[1], hh[2], hh[3],
                           lgs[0], lgs[1], lgs[2], lgs[3], lgs[amax], amax);
        }
    }

    ggml_backend_tensor_get(p.am, &token_out, 0, sizeof(int32_t));
    return true;
}
