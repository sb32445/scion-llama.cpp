#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// Scion MTP sidecar (release drafter lane).
//
// A small frozen-body drafter (fc1 / gelu / fc2 MLP, ~50 MB fp16 at the
// current width) that consumes the target's post-norm final hidden plus the
// raw token-embedding row of the token to be decoded, and proposes the token
// the target would produce next.  The final projection reuses the target
// model's own output_norm/output tensors in VRAM; the sidecar file carries
// only the two matrices (see moe/mtp_sidecar_export.py in the scion repo).
//
// Graph (per position):
//   e = tok_embd[token]                     (raw stored row)
//   x = concat(rms(h), rms(e))
//   z = fc2 @ gelu_erf(fc1 @ x)
//   logits = output @ (rms(z) * output_norm)
//   return argmax(logits)
//
// Reference implementation and acceptance/timing validation:
// tests/test-mtp-sidecar.cpp.
struct llama_mtp_sidecar {
    explicit llama_mtp_sidecar(const llama_model & model);
    ~llama_mtp_sidecar();

    llama_mtp_sidecar(const llama_mtp_sidecar &) = delete;
    llama_mtp_sidecar & operator=(const llama_mtp_sidecar &) = delete;

    // backends: the target context's backend list (CPU last), as required by
    // ggml_backend_sched_new.  Builds the head graph and uploads the sidecar
    // weights on the backends' default buffer types.
    bool load(const char * path, const std::vector<ggml_backend_t> & backends, std::string & err);

    // h: [n_embd_out] post-norm hidden at the row before `token`
    // token: the token to be decoded at the next position
    // token_out: the drafted token (argmax of the head's logits)
    bool draft(const float * h, int32_t token, int32_t & token_out, std::string & err);

    bool loaded() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
