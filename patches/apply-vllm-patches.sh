#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# Apply the vLLM patch stack in patches/vllm-<version>/ to the installed vLLM.
# docs/install-vllm-tt.sh runs this right after it installs vLLM; it can also be
# run by hand. The stack is the patch files listed, in order, in
# patches/vllm-<version>/series (paths relative to that directory; blank lines
# and # comments are skipped). Patches exist only for the pinned version; any
# other vLLM version is left untouched (exit 0), so a version bump silently
# drops them: see the README next to the series before bumping.
#
#   bash patches/apply-vllm-patches.sh [PYTHON]
#
# PYTHON is the interpreter whose vLLM is patched. It defaults to the one in
# $VIRTUAL_ENV (or ./.venv), the environment `uv pip install` just wrote to.
#
# Idempotent: the run finds how much of the stack is already applied and
# applies only the rest, so a re-run is a no-op. It checks the whole stack on a
# scratch copy of the files it touches before changing vLLM, and fails without
# touching anything when the installed files match no prefix of the stack (a
# partial apply, a locally edited file, a different vLLM build). Patches apply
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
# A VLLM_TARGET_DEVICE=empty build reports 0.31.0+empty; the patches follow the
# release, not the build target.
version = importlib.metadata.version("vllm").split("+")[0]
print(version, os.path.dirname(package))
EOF
)
read -r version site <<<"$info"

patch_dir=$patch_root/vllm-$version
if [[ ! -f $patch_dir/series ]]; then
    log "no patches for vLLM $version; nothing to apply"
    exit 0
fi
patches=()
while read -r entry || [[ -n $entry ]]; do
    entry=${entry%%#*}
    entry=${entry//[[:space:]]/}
    [[ -n $entry ]] || continue
    [[ -f $patch_dir/$entry ]] || { log "series lists $entry, which does not exist"; exit 1; }
    patches+=("$patch_dir/$entry")
done <"$patch_dir/series"
if ((${#patches[@]} == 0)); then
    log "the series for vLLM $version is empty; nothing to apply"
    exit 0
fi

n=${#patches[@]}
popts=(-p1 --fuzz=0 --no-backup-if-mismatch --batch --silent)
label() { echo "${1#"$patch_root"/}"; }

# A scratch copy of every file the stack touches, as installed.
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
while read -r rel; do
    if [[ -f $site/$rel ]]; then
        mkdir -p "$scratch/$(dirname "$rel")"
        cp -p "$site/$rel" "$scratch/$rel"
    fi
done < <(sed -nE 's#^(---|\+\+\+) [ab]/([^[:space:]]+).*#\2#p' "${patches[@]}" | sort -u)

# applied = the length of the stack prefix already in place: the topmost patch
# that reverses cleanly, after which every patch below it must reverse too.
applied=0
for ((i = n - 1; i >= 0; i--)); do
    if patch "${popts[@]}" -d "$scratch" -i "${patches[i]}" --reverse --forward --dry-run \
        >/dev/null 2>&1; then
        applied=$((i + 1))
        break
    fi
done
for ((i = applied - 1; i >= 0; i--)); do
    patch "${popts[@]}" -d "$scratch" -i "${patches[i]}" --reverse --forward >/dev/null 2>&1 || {
        log "$(label "${patches[i]}") is only partly applied to $site, or a later patch"
        log "in the stack was applied without it. Reinstall vLLM and re-run."
        exit 1
    }
done
# The scratch copy is now unpatched: the whole stack must apply to it.
for ((i = 0; i < n; i++)); do
    patch "${popts[@]}" -d "$scratch" -i "${patches[i]}" --forward >/dev/null 2>&1 || {
        log "$(label "${patches[i]}") does not apply to $site after the patches before it."
        log "Dry run against the installed files:"
        patch -p1 --fuzz=0 --batch -d "$site" -i "${patches[i]}" --forward --dry-run >&2 || true
        exit 1
    }
done

for ((i = 0; i < n; i++)); do
    if ((i < applied)); then
        log "$(label "${patches[i]}"): already applied to $site, skipping"
    else
        patch "${popts[@]}" -d "$site" -i "${patches[i]}" --forward
        log "$(label "${patches[i]}"): applied to $site"
    fi
done
