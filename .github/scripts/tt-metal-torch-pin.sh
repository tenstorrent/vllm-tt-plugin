# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# Print the x86_64 torch requirement that tt-metal pins at release tag $1.
# The ttnn wheel declares no torch, so this file is the only source of the pin.
set -euo pipefail
tag=$1
pin=$(curl -fsSL "https://raw.githubusercontent.com/tenstorrent/tt-metal/$tag/tt_metal/python_env/requirements-dev.txt" \
    | sed -nE "s/^(torch==[^ ;]+) *;.*x86_64.*/\1/p")
[ -n "$pin" ] || { echo "no x86_64 torch pin in tt-metal $tag requirements-dev.txt" >&2; exit 1; }
echo "$pin"
