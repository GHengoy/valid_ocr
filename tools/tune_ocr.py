#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_ocr.py — 날짜 인식 전처리 비교 (어떤 전처리가 인식률을 올리는지 실측)

포장기(machine 0)의 OK(ok==1) 이미지만 표본으로 쓴다. 이 경우 스탬프 날짜 =
세션 ex_date 이므로 정답이 확실하다. 각 전처리 변형으로 '날짜 영역'을 rec 인식해
정답과 일치하는 비율을 비교한다.

사용:
    python3 tune_ocr.py            # 표본 80장
    python3 tune_ocr.py 200        # 표본 200장
(앱이 GPU를 점유 중이면 엔진 로드가 실패하니, 앱을 잠시 끄고 실행하세요)
"""
import os
import re
import sys
import glob
import json
import random
import numpy as np
import cv2

# 프로젝트 루트(상위 폴더) 기준 + 모듈 import 경로
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from trt_ocr import TRTPaddleOCR, parse_stamp_text

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 80


def date_of(crop, eng):
    """crop 을 rec 인식 → 날짜 문자열(YYYY-MM-DD) 또는 ''."""
    txt = eng._recognize(crop)
    y, mo, d, _, _ = parse_stamp_text([txt])
    return (f"{y}-{mo}-{d}" if y else ""), txt


# ── 전처리 변형들 ──
def v_none(c):
    return c

def v_clahe(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def v_upscale2(c):
    return cv2.resize(c, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

def v_sharpen(c):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(c, -1, k)

def v_gray_stretch(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def v_clahe_up(c):
    return v_upscale2(v_clahe(c))

VARIANTS = [
    ("none", v_none),
    ("clahe", v_clahe),
    ("upscale2", v_upscale2),
    ("sharpen", v_sharpen),
    ("gray_stretch", v_gray_stretch),
    ("clahe+upscale", v_clahe_up),
]


def main():
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    db = cfg["sub_roi"]["date"]

    eng = TRTPaddleOCR(os.path.join(BASE, "onnx"))
    print("엔진:", eng.reason)
    if not eng.available:
        raise SystemExit("엔진 로드 실패 — 앱을 끄고(GPU 해제) 다시 실행하세요.")

    # 표본 수집: machine 0, ok==1
    items = []
    for folder in sorted(glob.glob(os.path.join(BASE, "date", "*"))):
        lj = os.path.join(folder, "list.json")
        if not os.path.exists(lj):
            continue
        s = json.load(open(lj, encoding="utf-8"))
        exp = f'{s["ex_date_year"]}-{s["ex_date_month"]}-{s["ex_date_day"]}'
        for n in range(14):
            try:
                if s["ok_list"][0][n] == 1:
                    p = os.path.join(folder, f"0_{n+1}.png")
                    if os.path.exists(p):
                        items.append((p, exp))
            except (IndexError, KeyError):
                pass
    random.Random(0).shuffle(items)
    items = items[:SAMPLE]
    print(f"표본: {len(items)}장 (포장기 OK 이미지)\n")

    score = {name: 0 for name, _ in VARIANTS}
    for p, exp in items:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        crop = img[int(db["y1"]*h):int(db["y2"]*h), int(db["x1"]*w):int(db["x2"]*w)]
        for name, fn in VARIANTS:
            try:
                rd, _ = date_of(fn(crop), eng)
            except Exception:
                rd = ""
            if rd == exp:
                score[name] += 1

    n = len(items)
    print("=== 전처리별 날짜 정확도 ===")
    for name, _ in VARIANTS:
        s = score[name]
        print(f"  {name:16s}: {100*s/n:5.1f}%  ({s}/{n})")
    best = max(score, key=score.get)
    print(f"\n최고: {best} ({100*score[best]/n:.1f}%)")


if __name__ == "__main__":
    main()
