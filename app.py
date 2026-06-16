#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
일부인 검증 시스템 - Flask 웹 기반
"""

from flask import Flask, render_template, request, jsonify, Response, send_file
from PIL import Image, ImageDraw, ImageFont
from pypylon import pylon
from time import localtime, strftime
import numpy as np
import cv2
import re
import os
import json
import time
import threading
import webbrowser
import shutil
from datetime import datetime, timedelta
from io import BytesIO

import db

app = Flask(__name__)

# ─── 설정 로드 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")   # 폰트/카메라설정(.pfs)/아이콘

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

config = load_config()

# ─── 전역 상태 ───
class AppState:
    def __init__(self):
        self.reset()
        self.cameras = None
        self.converter = None
        self.cam = None
        self.camera_ready = False
        self.latest_frame = None  # 최신 카메라 프레임 (BGR, 원본)
        self.latest_maked = None  # 크롭된 일부인 영역 (BGR, 스캔 시 RGB 변환)
        self.latest_mjpeg = None  # 미리 인코딩된 MJPEG JPEG 바이트
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()  # 새 프레임 알림
        self._cached_roi = None   # ROI 캐시
        self._cached_sub_roi = None  # sub_roi(날짜/호기 줄) 캐시
        self._cached_crop_coords = None  # 크롭 좌표 캐시
        self._cached_src_size = None     # 원본 해상도 캐시

    def reset(self):
        self.ex_date_year = ""
        self.ex_date_month = ""
        self.ex_date_day = ""
        self.ex_date = ""
        self.jo_ya = ""
        self.a_b = ""
        self.machine = 0  # 0=포장기, 1=멀티포장기
        self.ok_list = [[0] * 14 for _ in range(2)]
        self.date_list = [[""] * 14 for _ in range(2)]
        self.date_list[1][8] = ""   # 멀티포장기 9호기 없음
        self.date_list[1][12] = ""  # 멀티포장기 11.호기 없음
        self.date_list[1][13] = ""  # 멀티포장기 12.호기 없음
        self.b_c_list = [["gray"] * 14 for _ in range(2)]
        self.sign_list = [0, 0]
        self.folder_name = ""
        self.point = 0

state = AppState()


# ─── 카메라 ───
def load_camera():
    cfg = load_config()
    try:
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        cam_info = None
        for dev_info in devices:
            if dev_info.GetIpAddress() == cfg["camera_ip"]:
                cam_info = dev_info
                break
        if cam_info is None:
            print("카메라를 찾을 수 없습니다. IP:", cfg["camera_ip"])
            return False

        state.cameras = pylon.InstantCameraArray(1)
        state.cam = state.cameras[0]
        state.cam.Attach(tlFactory.CreateDevice(cam_info))
        state.cameras.Open()
        pylon.FeaturePersistence.Load(
            os.path.join(ASSETS_DIR, cfg["camera_setting"]),
            state.cam.GetNodeMap(), True
        )
        state.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        state.converter = pylon.ImageFormatConverter()
        state.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        state.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        state.camera_ready = True
        print("카메라 연결 성공:", cfg["camera_ip"])
        return True
    except Exception as e:
        print("카메라 연결 실패:", str(e))
        state.camera_ready = False
        return False


def get_cached_roi():
    """ROI를 캐싱하여 매 프레임 파일 I/O 방지"""
    if state._cached_roi is None:
        state._cached_roi = load_config().get("roi", {"x1": 240, "y1": 260, "x2": 580, "y2": 410})
    return state._cached_roi


def get_cached_sub_roi():
    """sub_roi(날짜/호기 줄 분수좌표)를 캐싱. 미설정이면 None."""
    if state._cached_sub_roi is None:
        state._cached_sub_roi = load_config().get("sub_roi") or {}
    return state._cached_sub_roi or None


def get_crop_coords(w, h):
    """크롭 좌표를 캐싱 (원본 해상도가 바뀌지 않는 한 재계산 안 함)"""
    if state._cached_crop_coords is not None and state._cached_src_size == (w, h):
        return state._cached_crop_coords
    cfg_roi = get_cached_roi()
    rx = w / 792.0
    ry = h / 607.0
    coords = (
        max(0, int(cfg_roi["x1"] * rx)),
        max(0, int(cfg_roi["y1"] * ry)),
        min(w, int(cfg_roi["x2"] * rx)),
        min(h, int(cfg_roi["y2"] * ry))
    )
    state._cached_crop_coords = coords
    state._cached_src_size = (w, h)
    return coords


def invalidate_roi_cache():
    """설정 변경 시 ROI 캐시 초기화"""
    state._cached_roi = None
    state._cached_crop_coords = None
    state._cached_sub_roi = None


def camera_capture_loop():
    """백그라운드 스레드에서 카메라 프레임 지속 캡처 (최적화)
    리사이즈+JPEG 인코딩은 CPU 부담이 크므로 목표 FPS(10)로 제한한다.
    카메라는 LatestImageOnly 라 계속 최신 프레임을 받으며, 처리만 솎아낸다."""
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 50]
    min_interval = 1.0 / 10.0   # 목표 10fps
    last_proc = 0.0
    while True:
        if not state.camera_ready:
            time.sleep(0.5)
            continue
        try:
            grabResult = state.cameras.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                now = time.time()
                if now - last_proc < min_interval:
                    grabResult.Release()      # FPS 초과분은 처리 생략(프레임 드롭)
                    continue
                last_proc = now
                image_bgr = state.converter.Convert(grabResult).GetArray()
                image_bgr = cv2.flip(image_bgr, -1)  # flip(-1) = rotate 180 (더 빠름)
                h, w = image_bgr.shape[:2]

                # ROI 크롭 (OCR용, BGR 상태로 저장 - 스캔 시 RGB 변환)
                crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_coords(w, h)
                maked_img = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

                # MJPEG용: 리사이즈 + 가이드 박스 + JPEG 인코딩
                cfg_roi = get_cached_roi()
                frame_small = cv2.resize(image_bgr, (792, 607), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(frame_small,
                              (cfg_roi["x1"], cfg_roi["y1"]),
                              (cfg_roi["x2"], cfg_roi["y2"]),
                              (0, 0, 255), 3)
                _, jpeg = cv2.imencode('.jpg', frame_small, encode_param)
                jpeg_bytes = jpeg.tobytes()

                with state.frame_lock:
                    state.latest_frame = image_bgr
                    state.latest_maked = maked_img
                    state.latest_mjpeg = jpeg_bytes
                state.frame_event.set()
            grabResult.Release()
        except Exception as e:
            print("카메라 캡처 오류:", str(e))
            time.sleep(0.5)


def backup_loop():
    """일정 주기로 sessions.db 를 backups/ 에 백업한다(온라인 백업, WAL 안전).
    config.json 의 backup_interval_hours(기본 24), backup_keep(기본 14) 사용."""
    backup_dir = os.path.join(BASE_DIR, "backups")
    while True:
        cfg = load_config()
        interval = max(1, int(cfg.get("backup_interval_hours", 24))) * 3600
        keep = int(cfg.get("backup_keep", 14))
        try:
            # min_interval 로 잦은 재시작 시 중복 백업 방지
            dest = db.backup_db(backup_dir, keep=keep, min_interval_sec=interval)
            if dest:
                print("DB 백업 완료:", os.path.basename(dest))
        except Exception as e:
            print("DB 백업 오류:", str(e))
        # 최대 1시간 간격으로 깨어나 '주기 경과 시에만' 백업 (긴 interval 도 정확히 맞춤)
        time.sleep(min(interval, 3600))


def generate_mjpeg():
    """MJPEG 스트림 - 새 프레임이 올 때만 즉시 전달 (Event 기반)"""
    while True:
        state.frame_event.wait(timeout=1)
        state.frame_event.clear()
        with state.frame_lock:
            jpeg_bytes = state.latest_mjpeg
        if jpeg_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')


# ─── OCR ───
# TensorRT(PaddleOCR PP-OCRv5) 엔진만 사용한다.
# 엔진은 app 시작 시 init_ocr_engine() 에서 1회 초기화한다.
# (엔진이 없거나 초기화 실패 시 OCR 은 동작하지 않고 '인식 실패'로 처리됨)
OCR_ENGINE = None


def init_ocr_engine():
    """TensorRT OCR 엔진 초기화. 실패하면 None (OCR 비활성)."""
    global OCR_ENGINE
    try:
        from trt_ocr import TRTPaddleOCR
        engine = TRTPaddleOCR(os.path.join(BASE_DIR, "onnx"))
        OCR_ENGINE = engine
        print("OCR 엔진(TensorRT):", engine.reason)
    except Exception as e:
        OCR_ENGINE = None
        print("OCR 엔진 초기화 실패 (OCR 비활성):", str(e))


# 날짜/호기/조 파싱은 trt_ocr.parse_stamp_text 로 단일화 (verify_ocr.py 와 공용)
from trt_ocr import parse_stamp_text as _parse_ocr_text  # noqa: E402


def img_ocr(maked_img, verbose=True):
    """일부인 영역 OCR → (년, 월, 일, 호기번호, 조). TensorRT(PaddleOCR) 전용."""
    if maked_img is None:
        return "", "", "", "", ""

    results = []
    raw_log = None  # 터미널 출력용
    engine_used = "none"
    if OCR_ENGINE is not None and OCR_ENGINE.available:
        try:
            if OCR_ENGINE.det is not None:
                # det(텍스트 검출) + rec: 위치·기울기·배경(바코드/영양정보)에 강건
                results = OCR_ENGINE.ocr(maked_img)
                engine_used = "TRT/det"
            else:
                # det 엔진 없을 때: 기울기 보정 + 줄 자동검출 휴리스틱
                results = OCR_ENGINE.ocr_stamp(maked_img)
                engine_used = "TRT"
            raw_log = results
        except Exception as e:
            print("TRT OCR 오류:", str(e))
            results = []

    parsed = _parse_ocr_text(results)
    y, mo, d, fac, jo = parsed

    # ─── 인식 결과를 터미널에 출력 ───
    if verbose:
        if isinstance(raw_log, dict):
            raw_str = "  ".join(f"[{k}] '{v}'" for k, v in raw_log.items())
        else:
            raw_str = "  ".join(f"'{t}'" for t in (raw_log or [])) or "(없음)"
        print(f"[OCR/{engine_used}] 인식원본: {raw_str}")
        print(f"        → 날짜={y}-{mo}-{d}  호기={fac or '?'}  조={jo or '?'}")

    return parsed


# ─── 불량/오인식 케이스 수집 (나중에 OCR 튜닝용) ───
BAD_CASES_DIR = os.path.join(BASE_DIR, "bad_cases")

def _save_bad_case(frames, expected_date, reason, machine, num, extra=None):
    """검증에서 인식이 어긋난 프레임을 통째로 저장해 둔다.
    이미지(프레임 전체)+메타(기대값·사유·프레임별 인식결과)를 남겨,
    나중에 'tools/review_bad_cases.py' 로 다시 돌려보며 튜닝할 수 있게 한다.

    reason: hogi_fail(호기 못읽음) | hogi_mismatch(선택≠인식, 명확한 오인식)
            date_fail(날짜 못읽음) | date_mismatch(날짜 다름; 오인식/진짜불량 혼재)
    """
    try:
        os.makedirs(BAD_CASES_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        case_dir = os.path.join(BAD_CASES_DIR, f"{ts}_{reason}_m{machine}n{num}")
        os.makedirs(case_dir, exist_ok=True)

        per_frame = []
        for i, (m, p) in enumerate(frames):
            fname = f"f{i+1}.png"
            try:
                cv2.imwrite(os.path.join(case_dir, fname), m)
            except Exception:
                fname = None
            y, mo, d, fac, jo = p
            per_frame.append({
                "frame": i + 1, "image": fname,
                "date": f"{y}-{mo}-{d}" if y else "",
                "hogi": fac or "", "jo": jo or "",
            })

        meta = {
            "timestamp": ts,
            "reason": reason,
            "machine": machine,
            "machine_name": "포장기" if machine == 0 else "멀티포장기",
            "num": num,
            "expected_date": expected_date,
            "selected_jo": state.a_b,
            "frames": per_frame,
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(case_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 보존: 최신 N개만 유지 (오래된 케이스 폴더 삭제)
        keep = int(config.get("bad_case_keep", 300))
        cases = sorted(
            (os.path.join(BAD_CASES_DIR, n) for n in os.listdir(BAD_CASES_DIR)),
            key=os.path.getmtime)
        for old in cases[:-keep] if keep > 0 else []:
            shutil.rmtree(old, ignore_errors=True)

        print(f"        🗂  불량 케이스 저장: bad_cases/{os.path.basename(case_dir)}")
    except Exception as e:
        print("불량 케이스 저장 오류:", str(e))


# ─── 검증 로직 ───
def valid_img(num, point=0):
    """일부인 검증 — 여러 프레임을 읽어 호기·날짜를 '다수결'로 결정한다.
    단발 촬영은 흔들림/흐림 한 프레임 때문에 호기 오기록·인식실패가 생기므로,
    수 프레임을 모아 합의된 값만 채택한다."""
    expected_date = f"{state.ex_date_year}-{state.ex_date_month}-{state.ex_date_day}"
    auto = (state.machine == 0 and num == 100)
    need_point_select = False

    # ── 여러 프레임 OCR ──
    frames = []  # [(maked_img, (y, mo, d, factory, jo))]
    for i in range(6):
        m = None
        with state.frame_lock:
            if state.latest_maked is not None:
                m = state.latest_maked.copy()
        if m is not None:
            frames.append((m, img_ocr(m, verbose=False)))
            # 조기 종료: '설정 날짜 + 동일 호기' 프레임 2개면 확신
            good = [p for (_m, p) in frames
                    if p[0] and f"{p[0]}-{p[1]}-{p[2]}" == expected_date and p[3].isdigit()]
            if len(good) >= 2 and len({p[3] for p in good}) == 1:
                break
        if i < 5:
            time.sleep(0.1)   # 다음 프레임(약 10fps) 대기

    if not frames:
        return {"success": False, "error": "카메라 이미지 없음"}

    valid = [(m, p) for (m, p) in frames if p[0]]   # 날짜가 읽힌 프레임만

    # 프레임별 인식 내용 (실패 시 터미널 디버그 출력용)
    frames_dump = " | ".join(
        f"#{i+1} {(p[0]+'-'+p[1]+'-'+p[2]) if p[0] else '날짜?'} 호기{p[3] or '?'} 조{p[4] or '?'}"
        for i, (_m, p) in enumerate(frames))

    # ── 호기 다수결 ──
    hogi_votes = {}
    for (_m, p) in valid:
        fac = p[3]
        if fac.isdigit() and 1 <= int(fac) <= 12:
            hogi_votes[fac] = hogi_votes.get(fac, 0) + 1
    factory = max(hogi_votes, key=hogi_votes.get) if hogi_votes else ""

    if auto:
        # 자동 모드: 다수결 호기로 셀 결정. 못 읽으면 기록하지 않음(엉뚱한 호기 방지)
        if not factory:
            print(f"[검증] ❌ 호기 인식 실패 → 기록 안 함 | {frames_dump}")
            _save_bad_case(frames, expected_date, "hogi_fail", state.machine, num)
            return {"success": False,
                    "error": "호기를 인식하지 못했습니다. 일부인을 맞추고 다시 시도하세요.",
                    "ocr_text": expected_date}
        actual_num = int(factory)
    else:
        actual_num = num

    # ── 날짜 다수결 ──
    date_votes = {}
    for (_m, p) in valid:
        date_votes[f"{p[0]}-{p[1]}-{p[2]}"] = date_votes.get(f"{p[0]}-{p[1]}-{p[2]}", 0) + 1
    majority_date = max(date_votes, key=date_votes.get) if date_votes else ""

    # 대표 프레임(이미지 저장용): 다수결 날짜를 가진 프레임 우선
    def _score(mp):
        _m, p = mp
        rec = f"{p[0]}-{p[1]}-{p[2]}" if p[0] else ""
        return (2 if rec and rec == majority_date else 0) + (1 if p[0] else 0)
    maked_img, rep = max(frames, key=_score)
    ocr_jo = rep[4]
    if majority_date:
        ocr_year, ocr_month, ocr_day = majority_date.split("-")
    else:
        ocr_year = ocr_month = ocr_day = ""
    ocr_date_str = majority_date if majority_date else "인식실패"

    # ── 호기 검증 (수동 모드): 인식 호기(다수결) ≠ 선택 호기 → 기록 안 함 ──
    stamp_hogi = {13: 11, 14: 12}.get(actual_num, actual_num)
    recognized_hogi = int(factory) if (factory and factory.isdigit()) else None
    if (not auto) and recognized_hogi is not None and recognized_hogi != stamp_hogi:
        print(f"[검증] ❌ 호기 불일치: 선택 {stamp_hogi}호기 / 인식 {recognized_hogi}호기 → 기록 안 함 | {frames_dump}")
        _save_bad_case(frames, expected_date, "hogi_mismatch", state.machine, actual_num,
                       extra={"selected_hogi": stamp_hogi, "recognized_hogi": recognized_hogi})
        return {
            "success": True, "result": "NG", "recorded": False,
            "num": actual_num,
            "reason": f"호기 불일치 — 선택 {stamp_hogi}호기 / 인식 {recognized_hogi}호기",
            "ocr_date": ocr_date_str, "expected_date": expected_date,
            "factory": factory, "ocr_jo": ocr_jo,
            "ok_count": state.ok_list[state.machine].count(1),
            "ng_count": state.ok_list[state.machine].count(2),
        }

    # ── 날짜 검증 ──
    if majority_date and majority_date == expected_date:
        state.ok_list[state.machine][actual_num - 1] = 1
        state.b_c_list[state.machine][actual_num - 1] = "green"
        result_status = "OK"
    else:
        state.ok_list[state.machine][actual_num - 1] = 2
        state.b_c_list[state.machine][actual_num - 1] = "red"
        result_status = "NG"
        # 날짜 NG 수집: 못 읽음(date_fail)은 명확한 오인식,
        # 다름(date_mismatch)은 오인식/진짜불량 혼재 → 사유로 구분해 둠
        _save_bad_case(frames, expected_date,
                       "date_fail" if not majority_date else "date_mismatch",
                       state.machine, actual_num,
                       extra={"recognized_date": ocr_date_str})

    machine_name = "포장기" if state.machine == 0 else "멀티포장기"
    mark = "✅" if result_status == "OK" else "❌"
    print(f"[검증] {mark} {machine_name} {actual_num}호기 | OCR(다수결,{len(frames)}프레임): "
          f"{ocr_date_str} | 설정: {state.ex_date} | 결과: {result_status}")
    if result_status == "NG":
        print(f"        프레임별: {frames_dump}")

    # NG 사유
    if result_status == "OK":
        reason = ""
    elif not majority_date:
        reason = "날짜를 인식하지 못했습니다"
    else:
        reason = "소비기한이 다릅니다"

    # 날짜 기록
    if ocr_year and ocr_month and ocr_day:
        state.date_list[state.machine][actual_num - 1] = ocr_date_str
    else:
        state.date_list[state.machine][actual_num - 1] = "인식실패"

    # 이미지 저장
    if state.folder_name:
        try:
            image = Image.fromarray(cv2.cvtColor(maked_img, cv2.COLOR_BGR2RGB))
            file_path = os.path.join(BASE_DIR, f"{state.folder_name}/{state.machine}_{actual_num}.png")
            image.save(file_path)
        except Exception as e:
            print("이미지 저장 오류:", str(e))

        save_session_json()

    ok_count = state.ok_list[state.machine].count(1)
    ng_count = state.ok_list[state.machine].count(2)

    # 인식된 조 vs 수동 선택한 조 비교 (불일치 시 경고용, 검증 결과엔 영향 없음)
    jo_mismatch = bool(ocr_jo and state.a_b and ocr_jo != state.a_b)

    return {
        "success": True,
        "recorded": True,
        "num": actual_num,
        "result": result_status,
        "date": state.date_list[state.machine][actual_num - 1],
        "color": state.b_c_list[state.machine][actual_num - 1],
        "ok_count": ok_count,
        "ng_count": ng_count,
        "need_point_select": need_point_select,
        "factory": factory,
        "ocr_jo": ocr_jo,             # 인식한 조
        "selected_jo": state.a_b,     # 선택한 조
        "jo_mismatch": jo_mismatch,   # 조 불일치(경고용, 기록은 됨)
        "ocr_date": ocr_date_str,        # 인식한 날짜
        "expected_date": expected_date,  # 설정한 소비기한
        "reason": reason                 # NG 사유
    }


# ─── 데이터 저장/로드 (SQLite) ───
# 세션 메타·검증 결과는 SQLite(db.py)에 저장한다. 플랫 JSON 파일과 달리
# 트랜잭션 단위 원자적 커밋이라 쓰기 중 정전에도 데이터가 깨지지 않는다.
# 이미지(PNG)는 기존처럼 date/<폴더>/ 에 그대로 저장한다.
def _current_session_dict():
    return {
        "ex_date_year": state.ex_date_year,
        "ex_date_month": state.ex_date_month,
        "ex_date_day": state.ex_date_day,
        "ex_date": state.ex_date,
        "jo_ya": state.jo_ya,
        "a_b": state.a_b,
        "machine": state.machine,
        "ok_list": state.ok_list,
        "date_list": state.date_list,
        "b_c_list": state.b_c_list,
        "sign_list": state.sign_list,
    }


def save_session_json():
    """현재 세션을 DB에 저장 (함수명은 호환을 위해 유지)."""
    if not state.folder_name:
        return
    db.save_session(state.folder_name, _current_session_dict())


def load_session_json(folder_name):
    """DB에서 세션 로드 → 기존 list.json 과 동일한 dict (없으면 None)."""
    return db.load_session(folder_name)


def get_date_folders():
    """세션 폴더명 목록(최신순)."""
    return db.list_folders()


def cleanup_old_data():
    """보관 기간이 지난 데이터 삭제 (DB + 이미지 폴더)."""
    cfg = load_config()
    retention_days = cfg.get("data_retention_days", 90)
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")

    deleted_folders = db.delete_before(cutoff)
    deleted = 0
    for folder in deleted_folders:
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
        deleted += 1
        print(f"오래된 데이터 삭제: {folder}")
    return deleted


# ─── Flask 라우트 ───

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def api_state():
    """현재 상태 반환"""
    return jsonify({
        "ex_date": state.ex_date,
        "ex_date_year": state.ex_date_year,
        "ex_date_month": state.ex_date_month,
        "ex_date_day": state.ex_date_day,
        "jo_ya": state.jo_ya,
        "a_b": state.a_b,
        "machine": state.machine,
        "ok_list": state.ok_list[state.machine],
        "date_list": state.date_list[state.machine],
        "b_c_list": state.b_c_list[state.machine],
        "sign_list": state.sign_list,
        "ok_count": state.ok_list[state.machine].count(1),
        "ng_count": state.ok_list[state.machine].count(2),
        "folder_name": state.folder_name,
        "camera_ready": state.camera_ready,
    })


@app.route('/api/set_date', methods=['POST'])
def api_set_date():
    """날짜 설정"""
    data = request.json
    state.ex_date_year = data.get("year", "")
    state.ex_date_month = data.get("month", "").zfill(2)
    state.ex_date_day = data.get("day", "").zfill(2)
    state.ex_date = f"{state.ex_date_year}년 {state.ex_date_month}월 {state.ex_date_day}일 까지"
    return jsonify({"success": True, "ex_date": state.ex_date})


@app.route('/api/set_shift', methods=['POST'])
def api_set_shift():
    """조/야, A/B조 설정 및 세션 초기화"""
    data = request.json
    state.jo_ya = data.get("jo_ya", "")
    state.a_b = data.get("a_b", "")

    # 폴더(이미지 저장용) 보장
    state.folder_name = f"date/{state.ex_date_year}-{state.ex_date_month}-{state.ex_date_day}_{state.jo_ya}_{state.a_b}"
    full_path = os.path.join(BASE_DIR, state.folder_name)
    os.makedirs(full_path, exist_ok=True)

    loaded = load_session_json(state.folder_name)
    if loaded:
        # 기존 세션 이어쓰기 (DB에서 로드)
        state.ok_list = loaded["ok_list"]
        state.date_list = loaded["date_list"]
        state.b_c_list = loaded["b_c_list"]
        state.sign_list = loaded["sign_list"]
    else:
        # 신규 세션 생성: 이전 세션의 인식정보(날짜/OK/NG)가 남지 않도록 초기화한 뒤 저장.
        # (이걸 빼면 직전 세션의 date_list/ok_list 가 그대로 새 날짜 세션에 저장되어,
        #  날짜를 바꿔도 인식정보가 그대로 보이는 문제가 생긴다)
        state.ok_list = [[0] * 14 for _ in range(2)]
        state.date_list = [[""] * 14 for _ in range(2)]
        state.b_c_list = [["gray"] * 14 for _ in range(2)]
        state.sign_list = [0, 0]
        save_session_json()

    state.machine = 0  # 기본 포장기
    return jsonify({"success": True, "folder_name": state.folder_name})


@app.route('/api/set_machine', methods=['POST'])
def api_set_machine():
    """포장기/멀티포장기 전환"""
    data = request.json
    state.machine = data.get("machine", 0)
    return jsonify({
        "success": True,
        "machine": state.machine,
        "ok_list": state.ok_list[state.machine],
        "date_list": state.date_list[state.machine],
        "b_c_list": state.b_c_list[state.machine],
        "ok_count": state.ok_list[state.machine].count(1),
        "ng_count": state.ok_list[state.machine].count(2),
    })


@app.route('/api/verify', methods=['POST'])
def api_verify():
    """일부인 검증"""
    data = request.json
    num = data.get("num", 100)
    point = data.get("point", 0)
    result = valid_img(num, point)
    return jsonify(result)


@app.route('/api/new_session', methods=['POST'])
def api_new_session():
    """새 일지 작성 (초기화)"""
    state.reset()
    return jsonify({"success": True})


@app.route('/api/history')
def api_history():
    """일지 내역 조회"""
    folders = get_date_folders()
    history = []
    for folder in folders:
        data = load_session_json(f"date/{folder}")
        if data:
            date_str = f"{data['ex_date_year']}-{data['ex_date_month']}-{data['ex_date_day']}"
            sign_ok = data.get("sign_list", [0, 0])
            sign_status = "O" if (sign_ok[0] == 1 and sign_ok[1] == 1) else "X"
            history.append({
                "folder": folder,
                "date": date_str,
                "ex_date": data.get("ex_date", ""),
                "jo_ya": data.get("jo_ya", ""),
                "a_b": data.get("a_b", ""),
                "sign_status": sign_status,
                "ok_count": [sum(1 for x in data["ok_list"][i] if x == 1) for i in range(2)],
                "ng_count": [sum(1 for x in data["ok_list"][i] if x == 2) for i in range(2)],
            })
    return jsonify(history)


@app.route('/api/history/<path:folder>')
def api_history_detail(folder):
    """특정 일지 상세 조회"""
    data = load_session_json(f"date/{folder}")
    if not data:
        return jsonify({"error": "데이터 없음"}), 404

    # 이미지 목록 확인
    folder_path = os.path.join(BASE_DIR, "date", folder)
    images = {}
    for m in range(2):
        for n in range(1, 15):
            img_name = f"{m}_{n}.png"
            if os.path.exists(os.path.join(folder_path, img_name)):
                images[f"{m}_{n}"] = True

    data["images"] = images
    data["folder"] = folder
    return jsonify(data)


@app.route('/api/history_image/<path:folder>/<filename>')
def api_history_image(folder, filename):
    """일지 이미지 조회"""
    img_path = os.path.join(BASE_DIR, "date", folder, filename)
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    return "", 404


@app.route('/api/sign', methods=['POST'])
def api_sign():
    """서명 저장"""
    data = request.json
    sign_index = data.get("index", 0)  # 0=품질, 1=생산
    sign_data = data.get("sign_data", "")  # base64 이미지
    folder = data.get("folder", state.folder_name)

    if not folder:
        return jsonify({"success": False, "error": "세션 없음"})

    # 폴더 키 정규화 ("date/<폴더명>" 형태로 통일)
    folder_key = folder if folder.startswith("date/") else f"date/{folder}"

    # base64 이미지 저장
    import base64
    if sign_data.startswith("data:image"):
        sign_data = sign_data.split(",", 1)[1]
    img_bytes = base64.b64decode(sign_data)
    img_path = os.path.join(BASE_DIR, folder_key, f"popup_button_19_sign_{sign_index}.png")
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    # 서명 상태 업데이트 (DB)
    db.set_sign(folder_key, sign_index)
    if folder_key == state.folder_name:
        state.sign_list[sign_index] = 1

    return jsonify({"success": True})


@app.route('/api/sign_image/<path:folder>/<int:index>')
def api_sign_image(folder, index):
    """서명 이미지 조회"""
    img_path = os.path.join(BASE_DIR, "date", folder, f"popup_button_19_sign_{index}.png")
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    return "", 404


@app.route('/settings')
def settings_page():
    return render_template('settings.html')


@app.route('/api/config', methods=['GET'])
def api_get_config():
    return jsonify(load_config())


@app.route('/api/config', methods=['POST'])
def api_set_config():
    data = request.json
    cfg = load_config()
    if "data_retention_days" in data:
        cfg["data_retention_days"] = int(data["data_retention_days"])
    if "camera_ip" in data:
        cfg["camera_ip"] = data["camera_ip"]
    if "font_scale" in data:
        cfg["font_scale"] = int(data["font_scale"])
    if "server_port" in data:
        cfg["server_port"] = int(data["server_port"])
    if "camera_setting" in data:
        cfg["camera_setting"] = data["camera_setting"]
    if "roi" in data:
        cfg["roi"] = data["roi"]
    if "sub_roi" in data:
        cfg["sub_roi"] = data["sub_roi"]
    save_config(cfg)
    invalidate_roi_cache()
    return jsonify({"success": True, "config": cfg})


@app.route('/api/snapshot')
def api_snapshot():
    """현재 카메라 프레임을 JPEG로 반환 (ROI 박스 없이)"""
    frame = None
    with state.frame_lock:
        if state.latest_frame is not None:
            frame = state.latest_frame.copy()
    if frame is None:
        return '', 204
    frame_resized = cv2.resize(frame, (792, 607))
    _, jpeg = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpeg.tobytes(), 200, {'Content-Type': 'image/jpeg'}


@app.route('/api/cleanup', methods=['POST'])
def api_cleanup():
    """오래된 데이터 수동 정리"""
    deleted = cleanup_old_data()
    return jsonify({"success": True, "deleted": deleted})


@app.route('/api/delete_history', methods=['POST'])
def api_delete_history():
    """일지 삭제 (비밀번호 확인)"""
    data = request.json
    folder = data.get("folder", "")
    password = data.get("password", "")

    cfg = load_config()
    if password != cfg.get("delete_password", "0000"):
        return jsonify({"success": False, "error": "비밀번호가 틀렸습니다."})

    folder_key = folder if folder.startswith("date/") else f"date/{folder}"
    folder_path = os.path.join(BASE_DIR, folder_key)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)
    db.delete_session(folder_key)  # DB 레코드도 삭제 (멱등)

    return jsonify({"success": True})


@app.route('/api/camera_toggle', methods=['POST'])
def api_camera_toggle():
    """카메라 ON/OFF 토글"""
    if state.camera_ready:
        # 카메라 끄기
        try:
            state.camera_ready = False
            time.sleep(0.3)
            if state.cameras is not None:
                state.cameras.StopGrabbing()
                state.cameras.Close()
                state.cameras = None
                state.cam = None
            print("카메라 OFF")
        except Exception as e:
            print("카메라 해제 오류:", e)
        return jsonify({"success": True, "camera_ready": False})
    else:
        # 카메라 켜기
        result = load_camera()
        print("카메라 ON" if result else "카메라 연결 실패")
        return jsonify({"success": result, "camera_ready": result})


@app.route('/api/restart', methods=['POST'])
def api_restart():
    """서버 재시작 - 현재 프로세스를 동일 인자로 다시 실행"""
    import sys
    print("서버 재시작 요청")
    def do_restart():
        # 카메라 해제 후 재시작
        try:
            state.camera_ready = False
            time.sleep(0.5)
            if state.cameras is not None:
                state.cameras.StopGrabbing()
                state.cameras.Close()
                state.cameras = None
                state.cam = None
                print("카메라 해제 완료")
        except Exception as e:
            print("카메라 해제 중 오류:", e)
        time.sleep(1)
        python = sys.executable
        os.execv(python, [python] + sys.argv)
    threading.Thread(target=do_restart, daemon=True).start()
    return jsonify({"success": True})


@app.route('/api/pfs_files')
def api_pfs_files():
    """assets/ 내 .pfs 파일 목록 반환"""
    pfs_files = []
    if os.path.isdir(ASSETS_DIR):
        for f in os.listdir(ASSETS_DIR):
            if f.lower().endswith('.pfs'):
                pfs_files.append(f)
    pfs_files.sort()
    return jsonify(pfs_files)


@app.route('/api/upload_pfs', methods=['POST'])
def api_upload_pfs():
    """PFS 파일 업로드"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "파일이 없습니다."})
    f = request.files['file']
    if not f.filename.lower().endswith('.pfs'):
        return jsonify({"success": False, "error": ".pfs 파일만 업로드 가능합니다."})
    os.makedirs(ASSETS_DIR, exist_ok=True)
    save_path = os.path.join(ASSETS_DIR, f.filename)
    f.save(save_path)
    return jsonify({"success": True, "filename": f.filename})


@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    """서버 종료 + 브라우저(웹페이지) 닫기.
    JS window.close() 는 브라우저 보안상 막히므로, 서버가 브라우저 프로세스를
    종료시켜 웹페이지를 닫는다. (응답을 먼저 보낸 뒤 백그라운드에서 처리)"""
    print("시스템 종료 요청")

    def _shutdown():
        time.sleep(0.6)  # 응답이 브라우저에 도달할 시간 확보
        # 카메라 정리
        try:
            state.camera_ready = False
            if state.cameras is not None:
                state.cameras.StopGrabbing()
                state.cameras.Close()
        except Exception:
            pass
        # 브라우저 종료(웹페이지 닫힘)
        os.system("pkill -f chromium; pkill -f chrome; pkill -f firefox")
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({"success": True})


@app.route('/api/print', methods=['POST'])
def api_print():
    """일지 인쇄"""
    try:
        import cups
        data = request.json
        folder = data.get("folder", state.folder_name)
        cfg = load_config()

        img_path = os.path.join(BASE_DIR, f"{folder}/list.png")
        if not os.path.exists(img_path):
            return jsonify({"success": False, "error": "인쇄할 일지 이미지가 없습니다"})

        conn = cups.Connection()
        conn.printFile(cfg["printer_name"], img_path, "Print Job", {})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def open_browser(url):
    """웹 UI를 크로미움 전체화면(터치 단말용)으로 띄운다.
    크로미움이 없으면 기본 브라우저(webbrowser)로 폴백한다.
    --start-fullscreen: 주소창/탭 없이 화면을 꽉 채워 시작(F11 로 해제 가능).
    완전 잠금형(키오스크)을 원하면 --start-fullscreen 을 --kiosk 로 바꾸면 된다."""
    import subprocess
    for binary in ("chromium-browser", "chromium", "google-chrome-stable", "google-chrome"):
        path = shutil.which(binary)
        if not path:
            continue
        try:
            subprocess.Popen([
                path,
                "--start-fullscreen",
                "--noerrdialogs",
                "--disable-infobars",
                "--disable-session-crashed-bubble",
                url,
            ])
            return
        except Exception as e:
            print("크로미움 실행 실패 → 기본 브라우저 폴백:", e)
            break
    webbrowser.open(url)


# ─── 시작 ───
if __name__ == "__main__":
    cfg = load_config()
    port = cfg.get("server_port", 5050)

    # DB 초기화 (요청 핸들러가 사용하므로 서빙 프로세스에서 무조건 초기화)
    db.init_db(os.path.join(BASE_DIR, "sessions.db"))

    # reloader 사용 시 자식 프로세스에서만 카메라/브라우저 실행 (부모는 감시만)
    is_reloader_child = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

    if is_reloader_child:
        # 기존 list.json → DB 이관 (1회, 멱등)
        migrated = db.migrate_from_json(BASE_DIR, cfg.get("data_dir", "date"))
        if migrated:
            print(f"list.json → DB 이관: {migrated}건")

        # OCR 엔진 초기화 (TensorRT 전용)
        init_ocr_engine()

        # DB 주기 백업 스레드
        threading.Thread(target=backup_loop, daemon=True).start()

        # 카메라 초기화
        camera_thread = threading.Thread(target=camera_capture_loop, daemon=True)
        camera_thread.start()
        load_camera()

        # 오래된 데이터 정리
        cleanup_old_data()

        # 브라우저 열기 (크로미움 전체화면)
        threading.Timer(1.5, lambda: open_browser(f"http://localhost:{port}")).start()

    # werkzeug 로그 (GET/POST) 숨기기
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(
        host=cfg.get("server_host", "0.0.0.0"),
        port=port,
        debug=False,
        use_reloader=True,
        threaded=True
    )
