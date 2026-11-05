#!/usr/bin/env bash
#
# Gets TRELLIS.2 for Apple Silicon (github.com/shivampkumar/trellis-mac) ready
# to make Cat Cam 3D heads: the exact versions Cat Cam's heads were made with,
# plus the fixes they needed. Afterwards run its own setup.sh; see
# docs/3d-heads.md for the whole walk-through.
#
#   tools/trellis/install.sh ~/trellis-mac
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:?usage: $0 <folder to put trellis-mac in>}"

fetch() {  # url folder commit: just that commit, no history; skipped if already there
    [ -d "$2/.git" ] && return
    echo "Fetching $(basename "$1" .git) ..."
    git init -q "$2"
    git -C "$2" fetch -q --depth 1 "$1" "$3"
    git -C "$2" checkout -q FETCH_HEAD
}

fetch https://github.com/shivampkumar/trellis-mac.git "$DIR" d58628f4f5b9c3de8274cb110074154f4b31cef2
cd "$DIR"
# setup.sh clones these at their newest; pinned here, it finds them and skips that.
fetch https://github.com/pedronaugusto/mtlbvh.git deps/mtlbvh fe6660bdb7de5a12ea00bd96df2e21bdcbdd40b1
fetch https://github.com/pedronaugusto/mtldiffrast.git deps/mtldiffrast c390a92aa296c5cf131f91253b8a11fe01ec11e4
fetch https://github.com/pedronaugusto/mtlmesh.git deps/mtlmesh 5e6749356ba39b15489c99487f55840d49923da7
fetch https://github.com/pedronaugusto/mtlgemm.git deps/mtlgemm c2e5d3fa15152bc4aef65773cab1ded5a1855d38
fetch https://github.com/pedronaugusto/trellis2-apple.git deps/trellis2-apple 698fb31113e2f79622c1f6df09cb2a03cf7da953
fetch https://github.com/EasternJournalist/utils3d.git deps/utils3d 9a4eb15e4021b67b12c460c7057d642626897ec8
# o-voxel builds against Eigen, a submodule a plain clone leaves out.
git -C deps/trellis2-apple submodule update -q --init --depth 1 o-voxel/third_party/eigen

# The Metal backends ask for C++17, which today's torch headers don't build with.
sed -i '' 's/-std=c++17/-std=c++20/g' deps/{mtlbvh,mtldiffrast,mtlmesh,mtlgemm}/setup.py \
    deps/trellis2-apple/o-voxel/setup.py
# TRELLIS_REMESH=1 switch, and texture baker fixes (black specks, dark seams, upside-down texture).
if git apply --check "$HERE/trellis-mac.patch" 2>/dev/null; then
    git apply "$HERE/trellis-mac.patch"
fi
cp "$HERE/run_local_dino.py" "$HERE/convert_dinov3.py" .

echo
echo "Ready. Next, in $DIR:"
echo "  bash setup.sh"
echo "  .venv/bin/python convert_dinov3.py dinov3-vitl16"
