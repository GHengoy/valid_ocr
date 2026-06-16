#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trt_ocr.py — PaddleOCR(PP-OCRv5) TensorRT 추론 래퍼 (Jetson Xavier NX / TRT 8.5)

- onnx/build_trt.sh 로 빌드한 korean_rec.trt (+선택 korean_det.trt) 엔진을 로드한다.
- det 엔진이 있으면  det(텍스트 영역 검출) → 각 줄 crop → rec(인식)  전체 파이프라인,
  없으면 입력 이미지를 한 줄로 보고 rec 만 수행한다.
- tensorrt / pycuda 가 없으면 available=False 가 되어 app.py 가 tesseract 로 폴백한다.

엔진/사전 위치 (기본): <project>/onnx/{korean_rec.trt, korean_det.trt, korean_dict.txt}
"""

import os
import re
import threading
import numpy as np
import cv2

# TensorRT / CUDA 런타임은 선택적 의존성 — 없으면 폴백
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    cuda.init()  # 컨텍스트는 TRTPaddleOCR 에서 primary context push/pop 으로 관리
    _TRT_IMPORT_OK = True
    _TRT_IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _TRT_IMPORT_OK = False
    _TRT_IMPORT_ERR = e


def normalize_jo(ch):
    """OCR로 읽은 조 문자 → "A조"/"B조" (인식 불가 시 "")."""
    if not ch:
        return ""
    c = ch.upper()
    if c in ("A", "4", "^"):      # A 의 흔한 오인식 보정
        return "A조"
    if c in ("B", "8", "ß"):      # B 의 흔한 오인식 보정
        return "B조"
    return ""


def _valid_md(mo, d):
    return 1 <= mo <= 12 and 1 <= d <= 31


def parse_stamp_text(results):
    """OCR 텍스트 리스트 → (년, 월, 일, 호기, 조). app.py / verify_ocr.py 공용.
    호기 코드 "2<조><호기>" 에서 호기(1~12)·조(A/B)를 추출. 조는 "A조"/"B조"/""."""
    ocr_year = ocr_month = ocr_day = factory = jo = ""

    # ── 날짜 ──
    # 1) YYYY[구분자]MM[구분자]DD  (월/일 유효성 검증)
    for text in results:
        m = re.search(r'(202\d)\s*[.\-\s﹒·,]\s*(\d{1,2})\s*[.\-\s﹒·,]\s*(\d{1,2})', text)
        if m and _valid_md(int(m.group(2)), int(m.group(3))):
            ocr_year = m.group(1)
            ocr_month = f"{int(m.group(2)):02d}"
            ocr_day = f"{int(m.group(3)):02d}"
            break
    # 2) "MM[구분자]DD 까지" + 연도(202x)  — 연도 뒤 노이즈 숫자(예: 20265.08.11) 대응
    if not ocr_year:
        year = ""
        for text in results:
            ym = re.search(r'202\d', text)
            if ym:
                year = ym.group(0)
                break
        if year:
            for text in results:
                m = re.search(r'(\d{1,2})\s*[.\-\s﹒·,]\s*(\d{1,2})\s*까지', text)
                if m and _valid_md(int(m.group(1)), int(m.group(2))):
                    ocr_year = year
                    ocr_month = f"{int(m.group(1)):02d}"
                    ocr_day = f"{int(m.group(2)):02d}"
                    break
    # 3) 연속 8자리
    if not ocr_year:
        for text in results:
            digits = ''.join(re.findall(r'\d+', text))
            m = re.search(r'(202\d)(\d{2})(\d{2})', digits)
            if m and _valid_md(int(m.group(2)), int(m.group(3))):
                ocr_year, ocr_month, ocr_day = m.group(1), m.group(2), m.group(3)
                break

    # ── 호기 + 조 ──
    # "2"(2공장 고정) + 조문자 + 호기(1~12). 조문자는 A/B 인데 OCR이 숫자 4/8 로
    # 오인식하는 경우가 많아 [ABab48] 로 한정한다. (연도 "202x" 등 숫자열 오매칭도 방지)
    for text in results:
        m = re.search(r'2\s*([ABab48])\s*(1[0-2]|[1-9])', text)
        if m:
            jo = normalize_jo(m.group(1))
            factory = m.group(2)
            break

    return ocr_year, ocr_month, ocr_day, factory, jo


class _TRTEngine:
    """단일 TensorRT 엔진 래퍼 (동적 shape, explicit batch)."""

    def __init__(self, engine_path, logger):
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"엔진 deserialize 실패(버전/아키텍처 불일치 가능): {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.n_bind = self.engine.num_bindings
        self.in_idx = [i for i in range(self.n_bind) if self.engine.binding_is_input(i)]
        self.out_idx = [i for i in range(self.n_bind) if not self.engine.binding_is_input(i)]

        # 디바이스 버퍼를 '최대 shape' 기준으로 1회만 선할당하고 매 추론에서 재사용한다.
        # (추론마다 cuda.mem_alloc/free 하면 Xavier NX 통합메모리에서 느리고
        #  NvMap ENOMEM(error 12) 을 유발 → 인식이 수 초로 느려짐)
        for i in self.in_idx:
            _mn, _op, mx = self.engine.get_profile_shape(0, i)
            self.context.set_binding_shape(i, mx)   # 출력 max shape 계산용
        self._dev = {}
        for idx in range(self.n_bind):
            shape = tuple(self.context.get_binding_shape(idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self._dev[idx] = cuda.mem_alloc(int(nbytes))

    def infer(self, inp):
        """inp: np.float32 NCHW → 출력 텐서 리스트 반환. (선할당 버퍼 재사용)"""
        inp = np.ascontiguousarray(inp.astype(np.float32))
        i = self.in_idx[0]
        self.context.set_binding_shape(i, inp.shape)
        cuda.memcpy_htod_async(self._dev[i], inp, self.stream)

        bindings = [int(self._dev[k]) for k in range(self.n_bind)]
        self.context.execute_async_v2(bindings, self.stream.handle)

        outs = []
        for o in self.out_idx:
            shape = tuple(self.context.get_binding_shape(o))
            dtype = trt.nptype(self.engine.get_binding_dtype(o))
            host = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh_async(host, self._dev[o], self.stream)
            outs.append(host)
        self.stream.synchronize()
        return outs


class TRTPaddleOCR:
    """PP-OCRv5 TensorRT OCR. ocr(bgr_img) → 인식된 텍스트 줄 리스트."""

    def __init__(self, model_dir, rec_h=48, rec_max_w=640, det_limit=960,
                 det_thresh=0.3, det_box_thresh=0.5):
        self.available = False
        self.reason = ""
        self.rec_h = rec_h
        self.rec_max_w = rec_max_w
        self.det_limit = det_limit
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        # CLAHE 대비 보정용 (흐릿한 일부인 스탬프 인식률 +12%p 측정됨)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if not _TRT_IMPORT_OK:
            self.reason = f"tensorrt/pycuda import 실패: {_TRT_IMPORT_ERR}"
            return

        rec_path = os.path.join(model_dir, "korean_rec.trt")
        det_path = os.path.join(model_dir, "korean_det.trt")
        dict_path = os.path.join(model_dir, "korean_dict.txt")

        if not os.path.exists(rec_path):
            self.reason = f"rec 엔진 없음: {rec_path} (onnx/build_trt.sh 실행 필요)"
            return
        if not os.path.exists(dict_path):
            self.reason = f"사전 없음: {dict_path}"
            return

        try:
            # Flask 요청 스레드에서 호출되므로 primary context 를 push/pop 으로 관리
            self.cuda_ctx = cuda.Device(0).retain_primary_context()
            self._lock = threading.Lock()
            self.cuda_ctx.push()
            try:
                self.logger = trt.Logger(trt.Logger.ERROR)
                self.rec = _TRTEngine(rec_path, self.logger)
                self.det = _TRTEngine(det_path, self.logger) if os.path.exists(det_path) else None
                self.chars = self._load_charset(dict_path)   # 워밍업 전에 사전 로드
                # 워밍업: 첫 추론의 cuDNN/cuBLAS 지연을 로드 시점에 미리 소진
                try:
                    self._recognize(np.zeros((self.rec_h, 120, 3), np.uint8))
                except Exception:
                    pass
            finally:
                self.cuda_ctx.pop()
            self.available = True
            self.reason = "OK (det+rec)" if self.det else "OK (rec only)"
        except Exception as e:
            self.reason = f"엔진 로드 실패: {e}"
            self.available = False

    # ── 사전 / CTC ──────────────────────────────────────────────
    def _load_charset(self, dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n").rstrip("\r") for ln in f]
        # PaddleOCR CTCLabelDecode: index 0 = blank, 그 뒤 dict, (옵션) 마지막 = space
        # 모델 출력 클래스 수에 맞춰 space 포함 여부를 런타임에 보정한다.
        return ["blank"] + lines  # space 는 _ctc_decode 에서 클래스 수 보고 추가

    def _ctc_decode(self, preds):
        """preds: [1, T, C] → 문자열. blank=0, 반복 제거."""
        seq = preds[0]
        num_classes = seq.shape[1]
        chars = self.chars
        # space 보정: 출력 C 가 len(chars)+1 이면 마지막 인덱스를 공백으로 처리
        space_idx = None
        if num_classes == len(chars) + 1:
            space_idx = num_classes - 1
        idxs = seq.argmax(axis=1)
        out = []
        last = -1
        for k in idxs:
            k = int(k)
            if k != 0 and k != last:
                if space_idx is not None and k == space_idx:
                    out.append(" ")
                elif k < len(chars):
                    out.append(chars[k])
            last = k
        return "".join(out).strip()

    # ── rec ─────────────────────────────────────────────────────
    def _rec_preprocess(self, img):
        # CLAHE 대비 보정: 흐릿한 스탬프 인식률 향상(측정상 +12%p). 그레이→CLAHE→3채널
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = self._clahe.apply(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        ratio = w / float(h) if h > 0 else 1.0
        rw = int(np.ceil(self.rec_h * ratio))
        rw = max(10, min(self.rec_max_w, rw))
        resized = cv2.resize(img, (rw, self.rec_h))
        x = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        x = (x - 0.5) / 0.5
        return x[np.newaxis, ...]

    def _recognize(self, line_img):
        if line_img is None or line_img.size == 0:
            return ""
        x = self._rec_preprocess(line_img)
        out = self.rec.infer(x)[0]
        if out.ndim == 2:  # [T, C] → [1, T, C]
            out = out[np.newaxis, ...]
        return self._ctc_decode(out)

    # ── 기울기 보정 + 줄 자동 검출 (일부인 전용) ──────────────────
    def _deskew(self, crop):
        """스탬프 텍스트 기울기를 추정해 수평으로 보정 (기울어진 숫자 오인식 방지)."""
        try:
            gray = self._clahe.apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            h, w = bw.shape
            ys, xs = np.where(bw[:int(h * 0.7)] > 0)   # 상단(스탬프) 위주
            if len(xs) < 50:
                return crop
            ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))[2]
            if ang < -45:
                ang += 90
            elif ang > 45:
                ang -= 90
            if abs(ang) < 1.5 or abs(ang) > 25:   # 거의 수평이거나 비정상 각도면 패스
                return crop
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
            return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderValue=(255, 255, 255))
        except Exception:
            return crop

    def _find_two_lines(self, crop):
        """수평 투영으로 스탬프 2줄(날짜/호기)의 (y1,y2) 밴드를 찾는다.
        위에서부터 가까운 밴드 최대 2개를 스탬프로 보고, 멀리 떨어진 영양정보 등은 제외."""
        gray = self._clahe.apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        h = bw.shape[0]
        rs = bw.sum(axis=1).astype(np.float32)
        if rs.max() <= 0:
            return None
        rs = np.convolve(rs, np.ones(3) / 3, "same")
        mask = rs > rs.max() * 0.15
        bands, s = [], None
        for i, v in enumerate(mask):
            if v and s is None:
                s = i
            elif not v and s is not None:
                if i - s >= h * 0.04:
                    bands.append([s, i])
                s = None
        if s is not None and h - s >= h * 0.04:
            bands.append([s, h])
        if not bands:
            return None
        stamp = [bands[0]]
        for b in bands[1:]:
            if len(stamp) < 2 and b[0] - stamp[-1][1] < h * 0.15:
                stamp.append(b)
            else:
                break
        if len(stamp) >= 2:
            d, f = stamp[0], stamp[1]
        else:
            a, b = stamp[0]
            mid = (a + b) // 2
            d, f = [a, mid], [mid, b]
        pad = int(h * 0.03)
        return ((max(0, d[0] - pad), min(h, d[1] + pad)),
                (max(0, f[0] - pad), min(h, f[1] + pad)))

    def ocr_stamp(self, bgr_img):
        """일부인 전용: 기울기 보정 → 2줄 자동 검출 → 각 줄 rec.
        반환: [날짜줄 텍스트, 호기줄 텍스트] (검출 실패 시 전체를 한 줄로)."""
        if not self.available or bgr_img is None or bgr_img.size == 0:
            return []
        with self._lock:
            self.cuda_ctx.push()
            try:
                img = self._deskew(bgr_img)
                lines = self._find_two_lines(img)
                if lines is None:
                    t = self._recognize(img)
                    return [t] if t else []
                out = []
                for (y1, y2) in lines:
                    if y2 > y1:
                        t = self._recognize(img[y1:y2])
                        if t:
                            out.append(t)
                return out
            except Exception as e:
                print("TRT OCR(stamp) 오류:", e)
                return []
            finally:
                self.cuda_ctx.pop()

    # ── det (DB) ────────────────────────────────────────────────
    def _det_preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self.det_limit / max(h, w), 1.0)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        nh = max(32, (nh // 32) * 32)
        nw = max(32, (nw // 32) * 32)
        resized = cv2.resize(img, (nw, nh))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        x = (resized.astype(np.float32) - mean) / std
        x = x.transpose(2, 0, 1)[np.newaxis, ...]
        return x, (w / nw, h / nh)

    def _det_boxes(self, img):
        x, (sx, sy) = self._det_preprocess(img)
        prob = self.det.infer(x)[0]          # [1,1,H,W]
        prob = np.squeeze(prob)
        seg = (prob > self.det_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        ih, iw = img.shape[:2]
        for cnt in contours:
            if cv2.contourArea(cnt) < 9:
                continue
            # box score(평균 확률) 필터
            mask = np.zeros(seg.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            if prob[mask == 1].mean() < self.det_box_thresh:
                continue
            xx, yy, ww, hh = cv2.boundingRect(cnt)
            # 약간 패딩(unclip 근사) 후 원본 좌표로 스케일
            pad = int(0.3 * hh)
            x1 = max(0, int((xx - pad) * sx))
            y1 = max(0, int((yy - pad) * sy))
            x2 = min(iw, int((xx + ww + pad) * sx))
            y2 = min(ih, int((yy + hh + pad) * sy))
            if x2 > x1 and y2 > y1:
                boxes.append((y1, x1, y2, x2))  # 위→아래 정렬용으로 y 우선
        boxes.sort(key=lambda b: (b[0], b[1]))
        return [(x1, y1, x2, y2) for (y1, x1, y2, x2) in boxes]

    # ── 공개 API ────────────────────────────────────────────────
    def ocr_regions(self, bgr_img, regions):
        """고정 레이아웃용: 분수좌표 sub-ROI 들을 각각 rec(한 줄) 인식.

        regions: { name: (x1, y1, x2, y2) }  좌표는 bgr_img 기준 0..1 분수.
        반환:    { name: text }  (위→아래 순서 유지)
        det 엔진 없이도 여러 줄을 줄 단위로 인식할 수 있다.
        """
        if not self.available or bgr_img is None or bgr_img.size == 0:
            return {}
        h, w = bgr_img.shape[:2]
        out = {}
        with self._lock:
            self.cuda_ctx.push()
            try:
                for name, fb in regions.items():
                    x1 = max(0, min(w, int(fb[0] * w)))
                    y1 = max(0, min(h, int(fb[1] * h)))
                    x2 = max(0, min(w, int(fb[2] * w)))
                    y2 = max(0, min(h, int(fb[3] * h)))
                    if x2 > x1 and y2 > y1:
                        out[name] = self._recognize(bgr_img[y1:y2, x1:x2])
                    else:
                        out[name] = ""
            except Exception as e:
                print("TRT OCR(region) 오류:", e)
            finally:
                self.cuda_ctx.pop()
        return out

    def ocr(self, bgr_img):
        """BGR 이미지 → 인식된 텍스트 줄 리스트(위→아래 순)."""
        if not self.available or bgr_img is None or bgr_img.size == 0:
            return []
        with self._lock:
            self.cuda_ctx.push()
            try:
                if self.det is not None:
                    lines = []
                    for (x1, y1, x2, y2) in self._det_boxes(bgr_img):
                        crop = bgr_img[y1:y2, x1:x2]
                        txt = self._recognize(crop)
                        if txt:
                            lines.append(txt)
                    return lines
                # det 없음: 전체를 한 줄로
                txt = self._recognize(bgr_img)
                return [txt] if txt else []
            except Exception as e:
                print("TRT OCR 오류:", e)
                return []
            finally:
                self.cuda_ctx.pop()
