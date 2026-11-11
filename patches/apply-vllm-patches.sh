#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# Apply patches/vllm-<version>/*.patch to the installed vLLM package.
# docs/install-vllm-tt.sh runs this right after it installs vLLM; it can also be
# run by hand. Patches exist only for the pinned version; any other vLLM version
# is left untouched (exit 0), so a version bump silently drops them: see
# patches/vllm-0.30.0/README.md before bumping.
#
#   bash patches/apply-vllm-patches.sh [PYTHON]
#
# PYTHON is the interpreter whose vLLM is patched. It defaults to the one in
# $VIRTUAL_ENV (or ./.venv), the environment `uv pip install` just wrote to.
#
# Idempotent: a patch that is already applied is skipped. A patch that applies
# neither forward nor in reverse (a partial apply, a locally edited file, a
# different vLLM build) fails the run without touching anything. Patches apply
# with no fuzz, so a hunk that has moved fails instead of landing in the wrong
# place.
set -euo pipefail

patch_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${1:-${VIRTUAL_ENV:-$PWD/.venv}/bin/python}
log() { echo "apply-vllm-patches: $*" >&2; }

[[ -x $python ]] || { log "no Python interpreter at $python"; exit 1; }
command -v patch >/dev/null || { log "the 'patch' tool is not installed"; exit 1; }

# find_spec and importlib.metadata locate vLLM without importing it: importing
# vllm pulls in torch and runs plugin discovery.
info=$("$python" - <<'EOF'
import importlib.metadata
import importlib.util
import os

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vllm is not installed for this interpreter")
package = list(spec.submodule_search_locations)[0]
# A VLLM_TARGET_DEVICE=empty build reports 0.30.0+empty; the patches follow the
# release, not the build target.
version = importlib.metadata.version("vllm").split("+")[0]
print(version, os.path.dirname(package))
EOF
)
read -r version site <<<"$info"

patch_dir=$patch_root/vllm-$version
shopt -s nullglob
patches=("$patch_dir"/*.patch)
if ((${#patches[@]} == 0)); then
    log "no patches for vLLM $version; nothing to apply"
    exit 0
fi

for p in "${patches[@]}"; do
    name=${p#"$patch_root"/}
    opts=(-p1 -d "$site" -i "$p" --fuzz=0 --no-backup-if-mismatch --batch)
    if patch "${opts[@]}" --reverse --forward --dry-run --silent >/dev/null 2>&1; then
        log "$name: already applied to $site, skipping"
    elif patch "${opts[@]}" --forward --dry-run --silent >/dev/null 2>&1; then
        patch "${opts[@]}" --forward --silent
        log "$name: applied to $site"
    else
        log "$name does not apply to $site, and is not already applied. Dry run:"
        patch "${opts[@]}" --forward --dry-run >&2 || true
        exit 1
    fi
done
