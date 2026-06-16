#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ocr.py — 저장된 date/ 이미지로 TRT OCR 인식 정확도 검증

앱과 동일한 파이프라인(sub_roi 영역별 rec + trt_ocr.parse_stamp_text)으로
기존에 저장된 일부인 이미지들을 다시 인식해 정확도를 측정한다.

Ground truth (정답):
  - 파일명 {machine}_{num}.png 의 num     → 호기 번호 (1~12)
  - 세션의 a_b                            → 조 (A조/B조)
  - 세션의 ex_date + ok_list==1(OK) 인 경우 → 스탬프 날짜는 ex_date 와 같아야 함
  (NG 이미지는 스탬프 날짜가 ex_date 와 다르므로 날짜 정확도 집계에서 제외)

주의: 호기/조 정답은 자동모드에서 옛 tesseract 결과로 저장됐을 수 있어
      완벽한 정답은 아니다(근사). 날짜(OK)가 가장 신뢰도 높은 지표.

사용: python3 verify_ocr.py
"""
import os
import re
import csv
import glob
import json
import cv2

from trt_ocr import TRTPaddleOCR, parse_stamp_text

BASE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(BASE, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_session(folder_path):
    """세션 정답 로드 (DB 우선, 없으면 list.json)."""
    # DB
    try:
        import db
        db.init_db(os.path.join(BASE, "sessions.db"))
        key = "date/" + os.path.basename(folder_path)
        s = db.load_session(key)
        if s:
            return s
    except Exception:
        pass
    # list.json
    p = os.path.join(folder_path, "list.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    cfg = load_config()
    sub = cfg.get("sub_roi")
    regions = ({k: (b["x1"], b["y1"], b["x2"], b["y2"]) for k, b in sub.items()}
               if sub else None)

    eng = TRTPaddleOCR(os.path.join(BASE, "onnx"))
    print("OCR 엔진:", eng.reason)
    if not eng.available:
        raise SystemExit("TRT 엔진을 사용할 수 없습니다. onnx/build_trt.sh 및 설치 확인.")

    # 기계별 집계: {0: [맞은수, 전체], 1: [..]}  (0=포장기, 1=멀티포장기)
    acc_date = {0: [0, 0], 1: [0, 0]}
    acc_hogi = {0: [0, 0], 1: [0, 0]}
    acc_jo = {0: [0, 0], 1: [0, 0]}
    n_img = 0
    mismatches = []
    rows = []  # CSV 상세

    folders = [f for f in sorted(glob.glob(os.path.join(BASE, "date", "*")))
               if os.path.isdir(f)]
    total_imgs = sum(len(glob.glob(os.path.join(f, "[01]_*.png"))) for f in folders)
    print(f"검증 대상: 폴더 {len(folders)}개 / 이미지 {total_imgs}장")
    print("(이미지가 많으면 수 분 걸립니다. 진행상황을 아래에 표시합니다)\n", flush=True)

    for fi, folder in enumerate(folders, 1):
        s = load_session(folder)
        if not s:
            continue
        n_in_folder = len(glob.glob(os.path.join(folder, "[01]_*.png")))
        print(f"[{fi}/{len(folders)}] {os.path.basename(folder)} "
              f"({n_in_folder}장)  누적 {n_img}/{total_imgs}", flush=True)
        exp_date = f'{s["ex_date_year"]}-{s["ex_date_month"]}-{s["ex_date_day"]}'
        exp_jo = s.get("a_b", "")

        for img_path in sorted(glob.glob(os.path.join(folder, "[01]_*.png"))):
            name = os.path.basename(img_path)
            m = re.match(r'([01])_(\d+)\.png$', name)
            if not m:
                continue
            machine, num = int(m.group(1)), int(m.group(2))
            try:
                ok = s["ok_list"][machine][num - 1]
            except (KeyError, IndexError, TypeError):
                ok = 0
            img = cv2.imread(img_path)
            if img is None:
                continue
            n_img += 1

            if regions:
                texts = [t for t in eng.ocr_regions(img, regions).values() if t]
            else:
                texts = eng.ocr(img)
            y, mo, d, fac, jo = parse_stamp_text(texts)
            rec_date = f"{y}-{mo}-{d}" if y else ""

            folder_name = os.path.basename(folder)
            rows.append([folder_name, name, ok, exp_date, rec_date,
                         num, fac, exp_jo, jo, " | ".join(texts)])

            # 날짜 (OK 이미지만)
            if ok == 1:
                acc_date[machine][1] += 1
                if rec_date == exp_date:
                    acc_date[machine][0] += 1
                elif len(mismatches) < 40:
                    mismatches.append((name, "날짜", exp_date, rec_date, texts))
            # 호기 (1~12)
            if 1 <= num <= 12:
                acc_hogi[machine][1] += 1
                if fac == str(num):
                    acc_hogi[machine][0] += 1
                elif len(mismatches) < 40:
                    mismatches.append((name, "호기", num, fac, texts))
            # 조
            if exp_jo:
                acc_jo[machine][1] += 1
                if jo == exp_jo:
                    acc_jo[machine][0] += 1
                elif len(mismatches) < 40:
                    mismatches.append((name, "조", exp_jo, jo, texts))

    # CSV 저장
    csv_path = os.path.join(BASE, "verify_ocr_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["folder", "image", "ok(1=OK,2=NG)", "expected_date", "ocr_date",
                    "expected_hogi", "ocr_hogi", "expected_jo", "ocr_jo", "ocr_text"])
        w.writerows(rows)

    def pct(a, b):
        return f"{100*a/b:5.1f}%  ({a:>4}/{b:<4})" if b else "    N/A"

    def both(acc):  # 합계
        return [acc[0][0] + acc[1][0], acc[0][1] + acc[1][1]]

    print(f"\n총 이미지: {n_img}장")
    print("=" * 64)
    print(f" 인식 정확도        {'전체':^16} {'포장기(0)':^16} {'멀티포장기(1)':^16}")
    print("=" * 64)
    for label, acc in [("날짜(OK)", acc_date), ("호기(1~12)", acc_hogi), ("조(A/B)", acc_jo)]:
        b = both(acc)
        print(f"  {label:10s}  {pct(*b):>16}  {pct(*acc[0]):>16}  {pct(*acc[1]):>16}")
    print("\n※ 멀티포장기(1)는 스탬프가 물리적으로 잘려 찍히는 경우가 많아 호기 인식이 낮음")
    print(f"\n상세 CSV: {csv_path}")

    if mismatches:
        print("\n--- 불일치 샘플 (최대 40) ---")
        for name, kind, exp, got, texts in mismatches[:40]:
            print(f"  [{kind}] {name}: 정답={exp!r} 인식={got!r}  OCR={texts}")


if __name__ == "__main__":
    main()
