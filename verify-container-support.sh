#!/usr/bin/env bash
# Check that this checkout can read Prism PQ2_0 GGUFs (type id 142) before
# spending 20 minutes on a build that cannot.
#
#   ./verify-container-support.sh [path/to/llama-cli]
#
# Exit code 0 means the source tree is the right one.

set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hdr="$root/ggml/include/ggml.h"
fail=0

echo "== tree =="
echo "path:   $root"
if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
    echo "branch: $(git -C "$root" rev-parse --abbrev-ref HEAD)"
    echo "commit: $(git -C "$root" rev-parse --short HEAD)"
else
    echo "branch: (not a git checkout)"
fi

echo
echo "== type table =="
if [ ! -f "$hdr" ]; then
    echo "FAIL: $hdr not found; is this a llama.cpp source tree?"
    exit 1
fi

pq_line="$(grep -nE '^[[:space:]]*GGML_TYPE_PQ2_0[[:space:]]*=' "$hdr" | head -1)"
count_line="$(grep -nE '^[[:space:]]*GGML_TYPE_COUNT[[:space:]]*=' "$hdr" | head -1)"

if [ -z "$pq_line" ]; then
    echo "FAIL: GGML_TYPE_PQ2_0 is not defined in ggml/include/ggml.h"
    echo "      This is upstream llama.cpp (or the 'master' branch of this fork)."
    echo "      Those builds stop at GGML_TYPE_COUNT = 43 and reject the model with"
    echo "      'invalid ggml type 142. should be in [0, 43)'."
    echo "      Fix: git clone -b moe-corr-runtime https://github.com/sky-is-green/prism-ml-llama.cpp"
    fail=1
else
    echo "OK:   ${pq_line#*:}"
fi

if [ -z "$count_line" ]; then
    echo "WARN: GGML_TYPE_COUNT not found (unexpected tree layout)"
else
    echo "info: ${count_line#*:}"
fi

echo
echo "== binary =="
bin="${1:-}"
if [ -n "$bin" ]; then
    if [ -x "$bin" ]; then
        echo "path:   $bin"
        "$bin" --version 2>&1 | head -2 | sed 's/^/        /'
    else
        echo "WARN: $bin is not executable yet; build it and re-run with the path"
    fi
else
    for cand in "$root/build/bin/llama-cli" "$root/build/bin/Release/llama-cli"; do
        if [ -x "$cand" ]; then
            echo "found:  $cand"
            echo "        run it by absolute path, or an older llama-cli on your PATH"
            echo "        will shadow it"
            break
        fi
    done
fi

echo
if [ "$fail" -ne 0 ]; then
    echo "RESULT: FAIL (this tree cannot load PQ2_0 GGUFs)"
    exit 1
fi
echo "RESULT: OK (this tree knows PQ2_0 = 142)"
echo
echo "Next: rm -rf build && cmake -B build -DGGML_CUDA=ON && \\"
echo "      cmake --build build -j --target llama-cli llama-server"
exit 0
