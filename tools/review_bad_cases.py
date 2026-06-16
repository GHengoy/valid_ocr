#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_bad_cases.py — 앱이 자동 수집한 불량/오인식 케이스 검토·재인식

앱(app.py)은 검증에서 인식이 어긋난 프레임을 bad_cases/ 에 통째로 모은다.
이 스크립트는 그 케이스들을 요약하고, '지금의' TRT OCR 엔진으로 다시 돌려
당시 인식 → 현재 인식 → 기대값을 나란히 보여 준다. 튜닝 전후 비교에 쓴다.

bad_cases/<시각>_<사유>_m<머신>n<호기>/
    f1.png ... fN.png   (검증 당시 캡처한 프레임들)
    meta.json           (기대값·사유·프레임별 당시 인식결과)

사유(reason):
    hogi_fail      — 호기 못 읽음 (자동모드)            → 오인식
    hogi_mismatch  — 선택 호기 ≠ 인식 호기              → 명확한 오인식
    date_fail      — 날짜 못 읽음                        → 오인식
    date_mismatch  — 날짜는 읽었으나 기대값과 다름        → 오인식/진짜불량 혼재

사용:
    python3 tools/review_bad_cases.py                  # 전체 요약 + 재인식
    python3 tools/review_bad_cases.py hogi_mismatch    # 특정 사유만
    python3 tools/review_bad_cases.py --no-rerun       # 재인식 없이 요약만
"""
import os
import sys
import json
import glob
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

BAD_DIR = os.path.join(BASE, "bad_cases")


def main():
    args = [a for a in sys.argv[1:]]
    rerun = "--no-rerun" not in args
    reasons = [a for a in args if not a.startswith("-")]

    if not os.path.isdir(BAD_DIR):
        raise SystemExit(f"수집된 케이스가 없습니다: {BAD_DIR}")

    cases = sorted(d for d in glob.glob(os.path.join(BAD_DIR, "*")) if os.path.isdir(d))
    if reasons:
        cases = [c for c in cases if any(f"_{r}_" in os.path.basename(c) for r in reasons)]
    if not cases:
        raise SystemExit("해당하는 케이스가 없습니다. (사유 필터 확인)")

    # 사유별 집계
    by_reason = {}
    for c in cases:
        meta_p = os.path.join(c, "meta.json")
        try:
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            r = meta.get("reason", "?")
        except Exception:
            r = "?"
        by_reason[r] = by_reason.get(r, 0) + 1

    print(f"수집 케이스: {len(cases)}개   (위치: bad_cases/)")
    for r, n in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  - {r:14s} {n}개")
    print()

    eng = None
    if rerun:
        from trt_ocr import TRTPaddleOCR, parse_stamp_text
        eng = TRTPaddleOCR(os.path.join(BASE, "onnx"))
        print("OCR 엔진:", eng.reason)
        if not eng.available:
            raise SystemExit("TRT 엔진 사용 불가 — 재인식 생략하려면 --no-rerun")
        print("=" * 70)

    for c in cases:
        name = os.path.basename(c)
        try:
            with open(os.path.join(c, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            print(f"[{name}] meta.json 읽기 실패 — 건너뜀")
            continue

        exp = meta.get("expected_date", "")
        extra = ""
        if meta.get("reason") == "hogi_mismatch":
            extra = f"  (선택 {meta.get('selected_hogi')}호기 / 당시인식 {meta.get('recognized_hogi')}호기)"
        elif meta.get("reason") == "date_mismatch":
            extra = f"  (당시인식 {meta.get('recognized_date')})"
        print(f"\n■ {name}")
        print(f"   사유={meta.get('reason')}  기대날짜={exp}  선택조={meta.get('selected_jo','')}{extra}")

        # 당시 프레임별 인식 (저장된 meta)
        for fr in meta.get("frames", []):
            print(f"     당시 #{fr['frame']}: 날짜={fr['date'] or '?'} 호기={fr['hogi'] or '?'} 조={fr['jo'] or '?'}")

        # 현재 엔진으로 재인식
        if eng is not None:
            for img_path in sorted(glob.glob(os.path.join(c, "f*.png"))):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                texts = eng.ocr(img)
                y, mo, d, fac, jo = parse_stamp_text(texts)
                rec = f"{y}-{mo}-{d}" if y else "?"
                fnum = os.path.basename(img_path)
                print(f"     지금 {fnum}: 날짜={rec} 호기={fac or '?'} 조={jo or '?'}  | OCR={texts}")

    print("\n끝. 튜닝 후 이 스크립트를 다시 돌려 '당시'와 '지금'을 비교하세요.")


if __name__ == "__main__":
    main()
