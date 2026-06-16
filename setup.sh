#!/bin/bash
# ============================================================
#  setup.sh — 일부인 검증 시스템 전체 설치 (한 번에)
#  대상: Jetson Xavier NX / JetPack 5.x (L4T R35.x) / TensorRT 8.5 / CUDA 11.4
#
#  설치 항목:
#    1) 시스템 패키지(apt)  : python3-pip, flask, pillow, numpy, opencv,
#                            cups, TensorRT 파이썬(python3-libnvinfer), tesseract
#    2) pip(--user)        : pytesseract, pycuda
#    3) Basler pypylon     : (Pylon SDK 설치 여부 확인 후 안내)
#    4) TensorRT OCR 엔진   : onnx/build_trt.sh 로 rec 엔진 빌드 (선택, ~수십 분)
#    5) 데이터 폴더(date/)
#
#  사용:
#    bash setup.sh              # 대화형 (TRT 빌드 여부 물어봄)
#    bash setup.sh --build-trt  # TRT 엔진 빌드까지 자동 수행
#    bash setup.sh --no-trt     # TRT 빌드 건너뜀
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_TRT="ask"
for a in "$@"; do
  case "$a" in
    --build-trt) BUILD_TRT="yes" ;;
    --no-trt)    BUILD_TRT="no" ;;
    *) echo "알 수 없는 옵션: $a"; exit 1 ;;
  esac
done

echo "=========================================="
echo " 일부인 검증 시스템 설치"
echo "=========================================="

# ── 1. 시스템 패키지 ──────────────────────────────────────────
echo ""
echo "[1/5] 시스템 패키지 설치 (apt)..."
sudo apt update -y
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    python3-flask \
    python3-pil \
    python3-numpy \
    python3-cups \
    python3-libnvinfer \
    tesseract-ocr \
    tesseract-ocr-kor \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libcups2-dev \
    cups

# opencv 는 이미 동작하면 건드리지 않는다 (JetPack/기존 cv2 가 apt 4.2 보다 최신일 수 있음)
if python3 -c "import cv2" 2>/dev/null; then
    echo "  -> cv2 이미 사용 가능 ($(python3 -c 'import cv2;print(cv2.__version__)' 2>/dev/null)) — opencv 설치 건너뜀"
else
    echo "  -> cv2 없음 → python3-opencv 설치"
    sudo apt install -y python3-opencv
fi

# ── 2. pip 패키지 (--user) ────────────────────────────────────
echo ""
echo "[2/5] pip 패키지 설치 (numpy, pytesseract, pycuda)..."
# pycuda 소스 빌드가 CUDA 헤더/라이브러리를 찾도록 경로 전달
#   (없으면 "fatal error: cuda.h: No such file or directory" 로 빌드 실패)
export PATH="/usr/local/cuda/bin:$PATH"                       # nvcc 탐지
export CUDA_ROOT="/usr/local/cuda"                            # pycuda 자동 인식
export CPATH="/usr/local/cuda/include${CPATH:+:$CPATH}"       # g++ 가 cuda.h 찾도록
export LIBRARY_PATH="/usr/local/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
# numpy 버전은 1.23.5 로 고정해야 한다 (양쪽 제약의 교집합):
#   - pycuda: numpy API >= 0x10 필요 → numpy >= 1.23 (1.17 이면 "compiled against
#             API 0x10 but numpy is 0xd" 로 import 실패)
#   - tensorrt 8.5.2 파이썬: 내부에서 np.bool 사용 → numpy < 1.24 필요
#             (1.24 는 np.bool 제거 → "module 'numpy' has no attribute 'bool'")
#   → 1.23.x 만 둘 다 만족. 1.23.5 = 1.23 마지막. (cv2 는 상위호환이라 무관)
python3 -m pip install --user "numpy==1.23.5"
python3 -m pip install --user pytesseract
# pycuda: 최신 버전은 Python 3.9+ 문법(str | Sequence[...])을 써서 Python 3.8 에서
# "TypeError: 'ABCMeta' object is not subscriptable" 로 import 실패한다.
# → Python 3.8 호환 버전으로 고정. (numpy 1.24.4 에 맞춰 빌드됨)
python3 -m pip install --user --force-reinstall --no-cache-dir "pycuda==2022.2.2"

# pycuda 가 의존성 mako 를 통해 최신 markupsafe(2.1+)를 끌어오면, 시스템 flask 1.1.1/
# jinja2 가 쓰는 soft_unicode 가 사라져 "cannot import name 'soft_unicode'" 로 앱이
# 안 뜬다. → soft_unicode 가 있는 markupsafe 2.0.1 로 되돌린다.(mako 와도 호환)
python3 -m pip install --user "markupsafe==2.0.1"

# ── 3. Basler pypylon / Pylon SDK ─────────────────────────────
echo ""
echo "[3/5] Basler 카메라(pypylon) 확인..."
if python3 -c "from pypylon import pylon" 2>/dev/null; then
    echo "  -> pypylon 사용 가능"
else
    echo "  pypylon 설치 시도..."
    if python3 -m pip install --user pypylon 2>/dev/null && python3 -c "from pypylon import pylon" 2>/dev/null; then
        echo "  -> pypylon 설치 완료"
    else
        echo "  !! pypylon 설치 실패 또는 Pylon SDK 미설치."
        echo "  !! Basler 카메라를 쓰려면 Pylon SDK(aarch64)를 먼저 설치하세요:"
        echo "  !!   https://www.baslerweb.com  →  pylon ... Linux ARM 64bit (aarch64) .deb"
        echo "  !!   sudo dpkg -i pylon_*.deb  후  python3 -m pip install --user pypylon"
        echo "  !! (카메라 없이 웹 UI만 띄우는 데는 영향 없음)"
    fi
fi

# ── 4. TensorRT OCR 엔진 빌드 ─────────────────────────────────
echo ""
echo "[4/5] TensorRT OCR 엔진..."
if [ -s "$SCRIPT_DIR/onnx/korean_rec.trt" ]; then
    echo "  -> rec 엔진 이미 있음 (onnx/korean_rec.trt) — 건너뜀"
else
    if [ "$BUILD_TRT" = "ask" ]; then
        echo "  rec 엔진(onnx/korean_rec.trt)이 없습니다. 지금 빌드하면 Xavier NX 기준"
        echo "  수십 분 걸릴 수 있습니다. (엔진은 이 기기 전용으로 1회만 빌드)"
        read -rp "  지금 빌드할까요? [y/N]: " ans
        [[ "$ans" == "y" || "$ans" == "Y" ]] && BUILD_TRT="yes" || BUILD_TRT="no"
    fi
    if [ "$BUILD_TRT" = "yes" ]; then
        bash "$SCRIPT_DIR/onnx/build_trt.sh"
    else
        echo "  -> 건너뜀. 나중에 빌드: bash onnx/build_trt.sh"
        echo "     (엔진이 없으면 OCR은 tesseract 로 폴백 동작)"
    fi
fi

# ── 5. 데이터 폴더 ────────────────────────────────────────────
echo ""
echo "[5/5] 데이터 폴더 확인..."
mkdir -p "$SCRIPT_DIR/date"

# ── 완료 ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " 설치 완료!"
echo "=========================================="
echo " 실행        : ./start.sh   (또는 bash start.sh)"
echo " 부팅 자동실행: ~/.config/autostart/valid-ocr.desktop (이미 등록됨)"
echo " 설정 파일    : config.json"
echo "=========================================="
