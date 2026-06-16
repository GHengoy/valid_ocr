#!/bin/bash
# ============================================================
#  start.sh — 일부인 검증 시스템 실행
#  사용: bash start.sh   또는   ./start.sh
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 기존 app.py 프로세스가 떠 있으면 종료
EXISTING=$(pgrep -f "python3 .*app\.py" 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "기존 프로세스 종료 (PID: $EXISTING)"
    kill $EXISTING 2>/dev/null
    sleep 2
    REMAIN=$(pgrep -f "python3 .*app\.py" 2>/dev/null)
    [ -n "$REMAIN" ] && kill -9 $REMAIN 2>/dev/null
fi

# venv 가 있으면 사용, 없으면 시스템 python3
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    PY="python"
else
    PY="python3"
fi

# pip --user 로 설치한 패키지(pytesseract, pycuda 등) 경로 보장
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"

echo "일부인 검증 시스템 시작 ($PY)"
exec "$PY" app.py
