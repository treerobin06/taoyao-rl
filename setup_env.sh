#!/bin/bash
# Linux GPU 服务器一键环境搭建（AutoDL / Vast.ai / 自建均可）
# macOS 不支持 MuJoCo + D4RL，请用 Linux
#
# Usage:
#   bash setup_env.sh
#
# 安装完成后：
#   source .venv/bin/activate
#   python smoke_test.py

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "  Taoyao RL · 环境搭建"
echo "  Project: $PROJECT_ROOT"
echo "=================================================="

# ---------- 1. 系统检查 ----------
echo ""
echo "[1/7] 系统检查"
uname -s | grep -q Linux || { echo "  ✗ 此脚本仅支持 Linux"; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $PY_VER"
case "$PY_VER" in
  3.10) ;;
  *) echo "  ✗ D4RL/mujoco-py 需要 Python 3.10（不支持 3.11），当前 $PY_VER"; exit 1 ;;
esac

# ---------- 2. 系统依赖 (MuJoCo 渲染需要的库) ----------
echo ""
echo "[2/7] 系统依赖（MuJoCo / mujoco-py 编译所需）"
if [ -x "$(command -v apt-get)" ]; then
  if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    libgl1-mesa-glx libegl1-mesa libosmesa6-dev \
    libglfw3 libglew2.2 patchelf \
    >/dev/null 2>&1 || echo "  ⚠ apt 安装部分失败，可能影响 MuJoCo 渲染"
  echo "  ✓ apt 系统依赖装好"
else
  echo "  ⚠ 非 Debian/Ubuntu 系统，请手动确保以下库存在："
  echo "    libGL libEGL libOSMesa libglfw libGLEW patchelf"
fi

# ---------- 3. MuJoCo 2.1 ----------
echo ""
echo "[3/7] MuJoCo 2.1（二进制，给 mujoco-py / D4RL 用）"
MUJOCO_DIR="$HOME/.mujoco/mujoco210"
MUJOCO_TARBALL="${MUJOCO_TARBALL:-/root/autodl-tmp/mujoco210-linux-x86_64.tar.gz}"
if [ -d "$MUJOCO_DIR" ]; then
  echo "  ✓ $MUJOCO_DIR 已存在"
else
  TMP_DIR="$(mktemp -d)"
  mkdir -p "$HOME/.mujoco"
  if [ -f "$MUJOCO_TARBALL" ]; then
    echo "  ✓ 使用本地缓存 $MUJOCO_TARBALL"
    cp "$MUJOCO_TARBALL" "$TMP_DIR/mujoco210-linux-x86_64.tar.gz"
  else
    echo "  下载 MuJoCo 2.1（国内机房可能较慢，失败可手动放到 $MUJOCO_TARBALL）"
    OK=0
    for URL in \
      "https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz" \
      "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz" \
      "https://gh-proxy.com/https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
    do
      echo "    trying $URL"
      if curl -L --connect-timeout 10 --max-time 120 --retry 2 --retry-delay 2 \
        "$URL" -o "$TMP_DIR/mujoco210-linux-x86_64.tar.gz"; then
        OK=1
        break
      fi
    done
    if [ "$OK" != "1" ]; then
      echo "  ✗ MuJoCo 下载失败。可在本机下载后上传到 $MUJOCO_TARBALL 再重跑。"
      rm -rf "$TMP_DIR"
      exit 1
    fi
  fi
  tar -xzf "$TMP_DIR/mujoco210-linux-x86_64.tar.gz" -C "$HOME/.mujoco"
  rm -rf "$TMP_DIR"
  echo "  ✓ MuJoCo installed to $MUJOCO_DIR"
fi
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$MUJOCO_DIR/bin:${LD_LIBRARY_PATH:-}"
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# ---------- 4. 创建 venv ----------
echo ""
echo "[4/7] 创建 venv"
if [ -d ".venv" ]; then
  echo "  ⚠ .venv 已存在，跳过创建（如需重建：rm -rf .venv 再跑此脚本）"
else
  python3 -m venv --system-site-packages .venv
  echo "  ✓ .venv created"
fi
source .venv/bin/activate
if ! grep -q "Taoyao RL: MuJoCo 2.1" .venv/bin/activate; then
  cat >> .venv/bin/activate <<'EOF'

# Taoyao RL: MuJoCo 2.1 for D4RL / mujoco-py
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
EOF
fi
if ! grep -q "Taoyao RL: GLIBCXX workaround" .venv/bin/activate; then
  cat >> .venv/bin/activate <<'EOF'

# Taoyao RL: GLIBCXX workaround for AutoDL conda libstdc++ shadowing system libstdc++
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
EOF
fi
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export PIP_INDEX_URL PIP_TRUSTED_HOST
pip install -q --timeout 60 -U pip setuptools wheel "Cython<3"

# ---------- 5. PyTorch (CUDA 11.8) ----------
echo ""
echo "[5/7] PyTorch (CUDA 11.8)"
if python - <<'PY'
import torch
assert torch.cuda.is_available()
major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
assert (major, minor) >= (2, 1)
print(f"  ✓ reuse existing torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
PY
then
  :
else
  echo "  installing torch==2.1.0+cu118"
  pip install -q --timeout 120 torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
fi

# ---------- 6. 其他依赖 ----------
echo ""
echo "[6/7] 其他依赖"
pip install -q --timeout 60 -r requirements.txt
pip install -q --timeout 60 --no-deps D4RL==1.1
if python -c "import mjrl" >/dev/null 2>&1; then
  echo "  ✓ mjrl import OK"
else
  echo "  installing mjrl（D4RL MuJoCo 注册依赖）"
  OK=0
  for MJRL_URL in \
    "git+https://github.com/aravindr93/mjrl@master" \
    "git+https://gh-proxy.com/https://github.com/aravindr93/mjrl@master"
  do
    if pip install -q --timeout 120 "$MJRL_URL"; then
      OK=1
      break
    fi
  done
  if [ "$OK" != "1" ]; then
    echo "  ✗ mjrl 安装失败。需要能访问 GitHub，或提前安装 mjrl。"
    exit 1
  fi
fi

# ---------- 7. 验证 ----------
echo ""
echo "[7/7] 验证"
export D4RL_SUPPRESS_IMPORT_ERROR=1
python -c "import torch; print(f'  ✓ torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import gym; print(f'  ✓ gym {gym.__version__}')"
python -c "import mujoco_py; print('  ✓ mujoco_py import OK')"
python -c "import d4rl; print('  ✓ d4rl import OK')" 2>&1 | tail -1

echo ""
echo "=================================================="
echo "  环境搭建完成"
echo ""
echo "  下一步："
echo "    source .venv/bin/activate"
echo "    python download_d4rl.py    # 预下载数据（5-10 min）"
echo "    python smoke_test.py       # 验证完整 pipeline (~3 min)"
echo "=================================================="
