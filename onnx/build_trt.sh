#!/bin/bash
# ============================================================
#  build_trt.sh — 로컬 PaddleOCR ONNX → TensorRT 엔진 빌드
#  대상: Jetson Xavier NX / JetPack 5.x (L4T R35.x) / TensorRT 8.5.x
#
#  - 같은 폴더의 korean_rec.onnx / korean_det.onnx / korean_dict.txt 사용
#    (GitHub 다운로드/prebuilt 로직 없음 — 외부 의존 X)
#  - TRT 엔진은 TensorRT 버전·GPU 아키텍처에 종속되므로
#    반드시 "이 기기에서" 빌드해야 한다. (다른 Jetson 엔진은 deserialize 실패)
#  - 일부인 검사는 고정 ROI 라서 기본은 인식(rec) 모델만 빌드한다.
#    검출(det)도 빌드하려면:   bash build_trt.sh --det
#  - 정밀도 FP32 고정: PP-OCRv5 det/rec 는 FP16 에서 출력이 NaN 이 되는
#    문제가 있어 FP16 을 쓰지 않는다.
#  - timeout 을 걸지 않는다. (Xavier NX 는 빌드가 수 분~십수 분 소요;
#    한 번만 빌드하면 이후엔 .trt 엔진을 재사용)
#
#  Usage:
#    bash build_trt.sh            # rec 만 빌드
#    bash build_trt.sh --det      # rec + det 빌드
#    bash build_trt.sh --det --swap   # det 빌드 시 임시 디스크 swap 6G 추가(권장)
# ============================================================
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
WS=512   # TensorRT workspace 메모리 캡 (MiB) — Xavier NX OOM 방지
SWAPFILE=/var/tmp/trt_build_swap

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

BUILD_DET=0
ADD_SWAP=0
for a in "$@"; do
  case "$a" in
    --det)  BUILD_DET=1 ;;
    --swap) ADD_SWAP=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo -e "${RED}알 수 없는 옵션: $a${RESET}"; exit 1 ;;
  esac
done

echo ""
echo -e "${BOLD}=== 로컬 ONNX → TensorRT 엔진 빌드 (Xavier NX / TRT 8.5) ===${RESET}"

# ── 1. trtexec 탐색 ───────────────────────────────────────────
TRTEXEC=""
for c in "$(command -v trtexec 2>/dev/null || true)" \
         /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec /usr/local/bin/trtexec; do
  if [ -n "$c" ] && [ -x "$c" ]; then TRTEXEC="$c"; break; fi
done
if [ -z "$TRTEXEC" ]; then
  echo -e "${RED}❌ trtexec 를 찾을 수 없습니다. (JetPack TensorRT 설치 확인: /usr/src/tensorrt/bin/trtexec)${RESET}"
  exit 1
fi
echo -e "  trtexec : ${CYAN}${TRTEXEC}${RESET}"
TRT_VER="$("$TRTEXEC" --version 2>&1 | grep -oE 'v[0-9]+' | head -1 || true)"
echo -e "  TensorRT: ${CYAN}${TRT_VER:-unknown}${RESET}"

# ── 2. ONNX 파일 확인 ─────────────────────────────────────────
need=( "$DIR/korean_rec.onnx" "$DIR/korean_dict.txt" )
[ "$BUILD_DET" = "1" ] && need+=( "$DIR/korean_det.onnx" )
for f in "${need[@]}"; do
  if [ ! -s "$f" ]; then
    echo -e "${RED}❌ 필요한 파일이 없습니다: $f${RESET}"
    exit 1
  fi
done

# 입력 텐서 이름 (paddle2onnx 기본은 'x'; onnx 모듈 있으면 정확히 추출)
input_name() {
  local f="$1"
  if python3 -c "import onnx" 2>/dev/null; then
    python3 -c "import onnx; print(onnx.load('$f').graph.input[0].name)" 2>/dev/null || echo x
  else
    echo x
  fi
}

# ── 3. (옵션) 임시 디스크 swap ────────────────────────────────
# Xavier NX 는 RAM 6.7GB + zram 만 있어 큰 빌드(det) 시 CUDA OOM 가능.
# --swap 지정 시 빌드 동안만 디스크 swap 6G 를 붙였다가 종료 시 제거한다.
cleanup_swap() {
  if [ "$ADD_SWAP" = "1" ] && swapon --show=NAME --noheadings 2>/dev/null | grep -q "$SWAPFILE"; then
    sudo swapoff "$SWAPFILE" 2>/dev/null || true
    sudo rm -f "$SWAPFILE" 2>/dev/null || true
    echo -e "  ${CYAN}임시 swap 제거됨.${RESET}"
  fi
}
trap cleanup_swap EXIT
if [ "$ADD_SWAP" = "1" ]; then
  if swapon --show=NAME --noheadings 2>/dev/null | grep -q "$SWAPFILE"; then
    echo -e "  임시 swap 이미 활성: $SWAPFILE"
  else
    echo -e "  ${YELLOW}임시 swap 6G 생성 (sudo 필요)...${RESET}"
    if sudo fallocate -l 6G "$SWAPFILE" 2>/dev/null || sudo dd if=/dev/zero of="$SWAPFILE" bs=1M count=6144 status=none; then
      sudo chmod 600 "$SWAPFILE"
      sudo mkswap "$SWAPFILE" >/dev/null
      sudo swapon "$SWAPFILE"
      echo -e "  ${GREEN}swap 추가됨.${RESET}"
    else
      echo -e "  ${YELLOW}⚠ swap 생성 실패 — swap 없이 진행${RESET}"
    fi
  fi
fi

# ── 4. 빌드 함수 ──────────────────────────────────────────────
# build <onnx> <engine> <shape옵션...>
build() {
  local onnx="$1" eng="$2"; shift 2
  echo ""
  echo -e "${BOLD}🔨 $(basename "$eng") 빌드${RESET} (FP32, workspace ${WS}MiB)"
  echo -e "  ${CYAN}(Xavier NX 는 수 분~십수 분 소요될 수 있습니다)${RESET}"
  local t0=$SECONDS
  if "$TRTEXEC" \
      --onnx="$onnx" \
      --saveEngine="$eng" \
      --memPoolSize=workspace:${WS} \
      --buildOnly "$@"; then
    if [ -s "$eng" ]; then
      echo -e "  ${GREEN}✅ 완료: $(basename "$eng")  ($((SECONDS - t0))초, $(du -h "$eng" | cut -f1))${RESET}"
      return 0
    fi
  fi
  echo -e "  ${RED}❌ 실패: $(basename "$eng")${RESET}"
  return 1
}

OK=1

# rec — 이미 빌드돼 있으면 건너뜀(재빌드 시간 낭비 방지). 다시 빌드하려면 onnx/korean_rec.trt 삭제 후 실행
REC_ONNX="$DIR/korean_rec.onnx"
if [ -s "$DIR/korean_rec.trt" ]; then
  echo -e "\n  ${CYAN}rec 엔진 이미 있음 → 건너뜀 (재빌드하려면 onnx/korean_rec.trt 삭제)${RESET}"
else
  REC_IN="$(input_name "$REC_ONNX")"
  echo -e "\n  rec 입력 텐서: ${CYAN}${REC_IN}${RESET}"
  build "$REC_ONNX" "$DIR/korean_rec.trt" \
    --minShapes=${REC_IN}:1x3x48x10 \
    --optShapes=${REC_IN}:1x3x48x320 \
    --maxShapes=${REC_IN}:1x3x48x640 || OK=0
fi

# det (옵션) — H/W 동적; 고정 ROI 라 max 960 으로 제한(원본 1920 대비 메모리↓)
if [ "$BUILD_DET" = "1" ]; then
  DET_ONNX="$DIR/korean_det.onnx"
  DET_IN="$(input_name "$DET_ONNX")"
  echo -e "\n  det 입력 텐서: ${CYAN}${DET_IN}${RESET}"
  build "$DET_ONNX" "$DIR/korean_det.trt" \
    --minShapes=${DET_IN}:1x3x64x64 \
    --optShapes=${DET_IN}:1x3x480x480 \
    --maxShapes=${DET_IN}:1x3x960x960 || OK=0
fi

# ── 5. 결과 ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}결과${RESET}"
echo -e "  사전 : ${DIR}/korean_dict.txt"
[ -s "$DIR/korean_rec.trt" ] && echo -e "  ${GREEN}rec  : ${DIR}/korean_rec.trt${RESET}" || echo -e "  ${RED}rec  : 없음${RESET}"
if [ "$BUILD_DET" = "1" ]; then
  [ -s "$DIR/korean_det.trt" ] && echo -e "  ${GREEN}det  : ${DIR}/korean_det.trt${RESET}" || echo -e "  ${RED}det  : 없음${RESET}"
fi
echo ""
if [ "$OK" = "1" ]; then
  echo -e "${GREEN}${BOLD}✅ TensorRT 엔진 빌드 완료.${RESET}"
else
  echo -e "${RED}${BOLD}일부 엔진 빌드 실패. 위 로그를 확인하세요.${RESET}"
  echo -e "${YELLOW}  det 빌드 중 OOM 이면 --swap 옵션을 붙여 다시 시도하세요.${RESET}"
  exit 1
fi
