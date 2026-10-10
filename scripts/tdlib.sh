#!/usr/bin/env bash
# Provision libtdjson on the host (SPEC §3): `scripts/tdlib.sh` installs the prebuilt
# library from the PyPI `tdjson` wheel; `scripts/tdlib.sh build` builds TDLib from source
# with CMake instead. Never copy a libtdjson built inside a container to the host: OpenSSL
# and libstdc++ ABI mismatches fail at load time.
#
# Both paths pin TDLib 1.8.67 (commit d1085f9c), the version tgi.td was written against,
# install to ${TDJSON_PATH:-$DATA_DIR/lib/libtdjson.so} and finish by loading the library
# and printing its version (a local td_execute call: no network, no account).
#
# The wheel (Linux x86_64, CPython 3.12) is third-party packaging of TDLib
# (https://github.com/AYMENJD/tdjson, MIT; TDLib itself is BSL-1.0). Its sha256 is pinned
# below. Only its bundled libtdjson is used, never its Python module.
#
# Build prerequisites (Debian/Ubuntu): sudo apt install git cmake g++ make gperf zlib1g-dev libssl-dev
# The build takes 30-60 min and is memory-hungry (SPEC §8).
set -euo pipefail

TDLIB_VERSION=1.8.67
TDLIB_COMMIT=d1085f9cebc5a62379991ae1652673954f229c1f
WHEEL=tdjson-1.8.67-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
WHEEL_URL=https://files.pythonhosted.org/packages/b0/8c/4910cb619990c717294993ae20a688303dada3913cf1f1912ee7d837e515/$WHEEL
WHEEL_SHA256=3dd10a16b2d0ee42fd12d05dcb7b0e0ba2f85f6dd6a8808054f7431e31ab5e36

mode=${1:-wheel}
case "$mode" in
    wheel | build) ;;
    *) echo "usage: scripts/tdlib.sh [wheel|build]" >&2; exit 1 ;;
esac
repo=$(cd "$(dirname "$0")/.." && pwd)

# DATA_DIR and TDJSON_PATH from the environment, else from .env (only these two keys are read).
env_value() {
    [ -f "$repo/.env" ] || return 0
    local v
    v=$(sed -n "s/^$1=//p" "$repo/.env" | tail -n 1)
    case $v in  # one pair of matching quotes, as `set -a; . ./.env` would strip
        \"*\") v=${v#\"}; v=${v%\"} ;;
        \'*\') v=${v#\'}; v=${v%\'} ;;
    esac
    printf '%s\n' "$v"
}
DATA_DIR=${DATA_DIR:-$(env_value DATA_DIR)}
TDJSON_PATH=${TDJSON_PATH:-$(env_value TDJSON_PATH)}
if [ -z "$TDJSON_PATH" ]; then
    [ -n "$DATA_DIR" ] || { echo "set DATA_DIR or TDJSON_PATH (in .env or the environment)" >&2; exit 1; }
    TDJSON_PATH=$DATA_DIR/lib/libtdjson.so
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

from_wheel() {
    if [ "$(uname -sm)" != "Linux x86_64" ]; then
        echo "the pinned wheel is Linux x86_64 only: use 'scripts/tdlib.sh build'" >&2
        exit 1
    fi
    curl -fsSL --proto '=https' -o "$work/$WHEEL" "$WHEEL_URL"
    echo "$WHEEL_SHA256  $work/$WHEEL" | sha256sum -c --quiet -
    uv run --no-project python - "$work/$WHEEL" "$work/libtdjson.so" <<'PY'
import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as wheel:
    (name,) = [n for n in wheel.namelist() if n.startswith("tdjson.libs/libtdjson")]
    with wheel.open(name) as src, open(sys.argv[2], "wb") as dst:
        shutil.copyfileobj(src, dst)
PY
}

from_source() {
    for tool in git cmake; do command -v "$tool" >/dev/null || { echo "build needs $tool (see the header)" >&2; exit 1; }; done
    git clone --quiet https://github.com/tdlib/td.git "$work/td"
    git -C "$work/td" checkout --quiet "$TDLIB_COMMIT"
    cmake -S "$work/td" -B "$work/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$work/build" --target tdjson -j "$(nproc)"
    cp -L "$work/build/libtdjson.so" "$work/libtdjson.so"
}

if [ "$mode" = wheel ]; then from_wheel; else from_source; fi

mkdir -p "$(dirname "$TDJSON_PATH")"
install -m 0644 "$work/libtdjson.so" "$TDJSON_PATH"
version=$(cd "$repo" && uv run python -c '
import sys
from pathlib import Path
from tgi.td import Hub, load_tdjson
print(Hub(load_tdjson(Path(sys.argv[1]))).version())' "$TDJSON_PATH")
echo "installed $TDJSON_PATH (TDLib $version)"
[ "$version" = "$TDLIB_VERSION" ] || { echo "expected TDLib $TDLIB_VERSION" >&2; exit 1; }
