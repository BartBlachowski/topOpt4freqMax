#!/usr/bin/env bash
# Reproducible setup of the SAND (Toebat & Feppon 2026) stack in the project .venv.
# Performed on 2026-10-08 on an Apple Silicon Mac (macOS 26, Python 3.13.2 venv).
#
#   bash tools/SAND/setup_sand_env.sh            # Python side only
#   bash tools/SAND/setup_sand_env.sh --freefem  # also install FreeFEM 4.15 into /Applications (macOS arm64)
#
# Why the odd steps:
#  * nullspace_optimizer 1.3.0 hard-depends on pypardiso, which needs Intel MKL.  MKL has no
#    arm64 macOS wheel, so the optimizer is installed with --no-deps and `pypardiso_shim.py`
#    (scipy SuperLU behind the same `spsolve` name) is dropped into site-packages.  The
#    optimizer only calls pypardiso for KKT systems of the 'linear_system' range-space method;
#    the paper uses method_xiC='qp', so results are unaffected.
#  * pymedit imports pyvista and pyperclip at import time although the 2D path never uses them.
#  * FreeFEM: Homebrew has no formula; releases 4.16/4.17 ship no macOS binary; 4.15 has an
#    Apple-Silicon .dmg whose binaries hard-code /Applications/FreeFem++.app/... dylib paths and
#    /opt/homebrew/opt/gcc (libgfortran), and arrive quarantined (SIGKILL, exit 137).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
test -x "$PY" || { echo "no .venv at $ROOT/.venv"; exit 1; }

"$PY" -m pip install --no-deps "nullspace_optimizer==1.3.0" pyfreefem pymedit
"$PY" -m pip install qpalm piqp osqp cvxopt colored sympy matplotlib pyvista pyperclip
SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
if ! "$PY" -c 'import pypardiso' 2>/dev/null; then
  cp "$ROOT/tools/SAND/pypardiso_shim.py" "$SP/pypardiso.py"
  echo "installed pypardiso shim -> $SP/pypardiso.py"
fi
"$PY" -c 'import nullspace_optimizer, pyfreefem, pymedit, qpalm, piqp, osqp; print("python side OK")'

if [[ "${1:-}" == "--freefem" ]]; then
  if [[ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]]; then echo "FreeFEM auto-install only scripted for macOS arm64"; exit 1; fi
  test -d /opt/homebrew/opt/gcc || brew install gcc            # libgfortran runtime
  if [[ ! -x /Applications/FreeFem++.app/Contents/ff-4.15.1/bin/FreeFem++ ]]; then
    TMP="$(mktemp -d)"
    curl -L -o "$TMP/ff.dmg" https://github.com/FreeFem/FreeFem-sources/releases/download/v4.15/FreeFEM-v4.15-Apple-Silicon-15.4.dmg
    hdiutil attach -nobrowse -readonly -mountpoint "$TMP/mnt" "$TMP/ff.dmg"
    cp -R "$TMP/mnt/FreeFem++.app" /Applications/        # /Applications is admin-writable, no sudo
    hdiutil detach "$TMP/mnt"
    chmod -R u+w /Applications/FreeFem++.app              # dmg files are 444; xattr needs write
    xattr -rc /Applications/FreeFem++.app                 # clear com.apple.quarantine
  fi
  echo 'cout << "FreeFem++ OK" << endl;' > /tmp/ff_hello.edp
  /Applications/FreeFem++.app/Contents/ff-4.15.1/bin/FreeFem++ -nw -ne /tmp/ff_hello.edp | tail -1
  echo 'add to PATH:  export PATH=/Applications/FreeFem++.app/Contents/ff-4.15.1/bin:$PATH   (sand_env.py finds it automatically)'
fi
