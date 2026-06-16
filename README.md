# 일부인 검증 시스템 (소비기한 OCR 검사)

포장 제품의 **일부인 스탬프(소비기한 + 호기/작업자)** 를 카메라로 찍어 OCR로 읽고,
설정한 소비기한과 **일치하는지 자동 검증**하는 Flask 웹 기반 현장 단말 프로그램입니다.

- **OCR**: PaddleOCR PP-OCRv5(한국어) → ONNX → **TensorRT** 가속
- **UI**: 브라우저 풀스크린(키오스크), 터치 조작
- **데이터**: SQLite(`sessions.db`) + 일정 주기 자동 백업
- **대상 하드웨어**: NVIDIA **Jetson Xavier NX** / JetPack 5.x (L4T R35.x, TensorRT 8.5, CUDA 11.4, Python 3.8)

---

## 빠른 시작

```bash
cd ~/valid_ocr
bash scripts/setup.sh          # 의존성 설치 + (선택)엔진 빌드 + 바탕화면 아이콘 + 자동실행 등록
bash scripts/start.sh             # 실행
```
- 부팅 시 자동 실행되도록 등록됩니다(자동로그인 → GNOME → 자동 시작).
- 바탕화면 아이콘: **일부인 검증 시스템 / Jetson 재시작 / Jetson 종료**

웹 UI: `http://localhost:5050` (앱이 자동으로 브라우저를 엽니다)

---

## 설치 (setup.sh)

`bash scripts/setup.sh` 가 한 번에 처리합니다:

1. **시스템 패키지(apt)** — flask, pillow, opencv, cups, `python3-libnvinfer`(TensorRT) 등
2. **pip(--user)** — 버전 고정 필수:
   - `numpy==1.23.5` · `markupsafe==2.0.1` · `pycuda==2022.2.2`
3. **Basler pypylon** — Pylon SDK 확인/설치 안내
4. **TensorRT OCR 엔진** — `onnx/build_trt.sh` 로 rec 엔진 빌드(이 기기 전용, 수십 분)
5. **데이터 폴더**(`date/`)
6. **바탕화면 아이콘 + 부팅 자동실행** 등록

옵션: `bash scripts/setup.sh --build-trt`(엔진까지 자동) / `--no-trt`(엔진 건너뜀)

> ⚠️ Jetson 의존성은 버전이 까다롭습니다. 반드시 `setup.sh` 로 설치하세요(아래 "문제 해결" 참고).

### Basler 카메라
`config.json` 의 `camera_ip` 를 카메라 IP로 맞추고, Pylon SDK(aarch64)를 설치해야 합니다.
카메라가 없어도 웹 UI는 뜹니다(스캔만 불가).

### TensorRT 엔진 빌드
엔진(`onnx/korean_rec.trt`)은 **TensorRT 버전·GPU 아키텍처에 종속**되어 반드시 이 기기에서 빌드해야 합니다.
```bash
bash onnx/build_trt.sh           # rec 만 (권장)
bash onnx/build_trt.sh --swap    # 빌드 중 OOM 시 임시 디스크 swap 추가
```
엔진(`onnx/korean_rec.trt`)이 없으면 OCR이 동작하지 않아 스캔이 모두 '인식 실패'로 처리됩니다.

---

## 실행 / 종료

| 동작 | 방법 |
|---|---|
| 실행 | 바탕화면 **일부인 검증 시스템** 아이콘 또는 `bash scripts/start.sh` |
| 부팅 자동실행 | 설치 시 자동 등록(`~/.config/autostart/valid-ocr.desktop`) |
| 프로그램 종료 | 웹 UI 우하단 **종료** 버튼(앱+브라우저만 종료, 보드는 안 꺼짐) |
| 보드 재시작 | 바탕화면 **Jetson 재시작** 아이콘 |
| 보드 종료 | 바탕화면 **Jetson 종료** 아이콘 |

`bash scripts/start.sh` 는 터미널에서 실행 시 스캔할 때마다 인식 로그를 출력합니다:
```
[OCR/TRT] 인식원본: [date] '2026.08.11 까지'  [factory] '안성 2A1 남하진 0601'
        → 날짜=2026-08-11  호기=1  조=A조
```

---

## 일부인 스탬프 / 검증 규칙

스탬프는 2줄 고정 레이아웃:
```
2026.08.11 까지          ← 소비기한
안성 2B1  홍새롬 1943     ← 호기 코드 + 작업자
```
호기 코드 `2<조><호기>`: 앞 `2`=2공장(고정), 가운데 `A/B`=조(12h 교대, **수동 선택**), 마지막=**호기(1~12)**.

**검증 동작**
- **소비기한**: OCR 날짜 = 설정 날짜 → OK(녹색) / 다르면 NG(빨강)
- **호기**: 인식 호기 ≠ 선택 호기 → **기록 안 함**(잘못 스캔 방지). 못 읽으면 수동 선택 신뢰
- **조**: 인식률이 낮아 차단하지 않고, 다르면 **노란 경고**만(선택한 조로 기록)
- 자동 호기 인식은 **포장기에서만**. 멀티포장기는 호기가 스탬프에 잘 안 찍혀 수동 입력

NG 시 빨간 화면에 **인식한 글자 + 사유**(소비기한 불일치 / 호기 불일치 등)가 표시됩니다.

---

## 데이터 저장 / 백업

- 세션·검증 결과: **SQLite `sessions.db`** (트랜잭션·WAL → 정전에도 안 깨짐)
- 캡처 이미지: `date/<날짜_조_조명>/<기계>_<호기>.png`
- **자동 백업**: `backups/sessions_YYYYMMDD_HHMMSS.db` (온라인 백업 API)
  - 주기/보관: `config.json` 의 `backup_interval_hours`(기본 24), `backup_keep`(기본 14)
  - 복원: 앱 종료 후 `cp backups/<파일>.db sessions.db && rm -f sessions.db-wal sessions.db-shm`

> 과거 `list.json` 방식은 SQLite로 이관 후 폐기되었습니다.

### 불량/오인식 케이스 자동 수집 (OCR 튜닝용)

검증에서 인식이 어긋난 프레임은 `bad_cases/` 에 자동 저장됩니다(나중에 OCR 튜닝/회귀 비교용).

- 저장 단위: `bad_cases/<시각>_<사유>_m<기계>n<호기>/` (프레임 `f1.png…` + `meta.json`)
- 사유: `hogi_fail`(호기 못읽음) · `hogi_mismatch`(선택≠인식, **명확한 오인식**) · `date_fail`(날짜 못읽음) · `date_mismatch`(날짜 다름 — 오인식/진짜불량 혼재)
- 보존: 최신 `bad_case_keep`개(기본 300)만 유지
- 검토/재인식: `python3 tools/review_bad_cases.py [사유]` — 당시 인식 vs 지금 엔진 인식 vs 기대값 비교

---

## 설정 (config.json)

| 키 | 설명 |
|---|---|
| `camera_ip` | Basler 카메라 IP |
| `camera_setting` | Pylon 카메라 설정(.pfs) 파일명 |
| `server_port` | 웹 서버 포트(기본 5050) |
| `data_retention_days` | 데이터 보관 일수(초과 시 자동 삭제) |
| `delete_password` | 일지 삭제 비밀번호 |
| `roi` | 전체 일부인 영역(792×607 미리보기 좌표) |
| `sub_roi` | 날짜 줄 / 호기 줄 영역(ROI 크롭 기준 0~1 분수) |
| `backup_interval_hours` / `backup_keep` | DB 백업 주기 / 보관 개수 |
| `bad_case_keep` | 불량/오인식 케이스 보관 개수(기본 300) |

웹 UI의 **설정(⚙)** 페이지에서도 ROI·카메라·포트 등을 조정할 수 있습니다.

---

## 파일 구성

```
app.py                  Flask 서버 (라우트, 검증 로직, 카메라/MJPEG, 백업 스레드)
trt_ocr.py              TensorRT OCR 엔진 래퍼 + 텍스트 파싱(parse_stamp_text)
db.py                   SQLite 저장소 + 백업(backup_db)
config.json             설정
requirements.txt
templates/              index.html(메인 UI), settings.html(설정)
onnx/                   모델·엔진
  build_trt.sh          ONNX → TensorRT 엔진 빌드 (이 기기에서 1회)
  korean_rec.onnx / korean_det.onnx / korean_dict.txt / korean_rec.trt
scripts/                실행/설치 스크립트
  setup.sh              전체 설치 (의존성·아이콘·자동실행·polkit)
  start.sh              실행 런처
  jetson_reboot.sh / jetson_shutdown.sh   보드 재시작/종료(확인창)
tools/                  개발/점검 도구
  verify_ocr.py         저장 이미지로 인식 정확도 측정(기계별)
  tune_ocr.py           전처리 비교 도구(어떤 전처리가 좋은지 실측)
  review_bad_cases.py   수집된 불량/오인식 케이스 검토·재인식
assets/                 자산
  icons/                바탕화면 아이콘(SVG)
  *.pfs                 Pylon 카메라 설정
  malgun.ttf            폰트
(런타임, git 제외) date/  sessions.db  backups/
```

---

## OCR 파이프라인

1. 카메라 프레임 → `roi` 크롭(`maked_img`)
2. `sub_roi` 로 **날짜 줄 / 호기 줄** 분리 (det 불필요)
3. 각 줄을 rec 엔진으로 인식 — 입력에 **CLAHE 대비 보정**(흐릿한 스탬프 +12%p)
4. `parse_stamp_text` 정규식으로 날짜/호기/조 추출 (조 문자 A↔4·B↔8 오인식 보정, 날짜 "까지" 앵커 + 유효성 검증)

**정확도(저장 데이터 기준)**: 포장기 — 날짜/호기/조 ~85–87%. 멀티포장기 — 날짜 ~91%(호기는 스탬프 미인쇄로 낮음 → 수동).
정확도는 `python3 tools/verify_ocr.py` 로 기계별 재측정할 수 있습니다.

---

## 문제 해결 (Jetson 의존성)

`setup.sh` 가 모두 처리하지만, 수동 대응이 필요할 때 참고:

| 증상 | 원인 / 해결 |
|---|---|
| `pycuda ... module compiled against API 0x10 but numpy is 0xd` | numpy가 너무 낮음 → `pip install --user numpy==1.23.5` |
| `module 'numpy' has no attribute 'bool'` | numpy 1.24+ 가 tensorrt와 충돌 → **numpy==1.23.5** (1.24 아님) |
| `pycuda: 'ABCMeta' object is not subscriptable` | 최신 pycuda가 Py3.9 문법 사용 → `pycuda==2022.2.2` |
| `fatal error: cuda.h` (pycuda 빌드) | CUDA 경로 전달: `export CUDA_ROOT=/usr/local/cuda CPATH=/usr/local/cuda/include PATH=/usr/local/cuda/bin:$PATH` |
| `cannot import name 'soft_unicode'` | markupsafe 너무 높음 → `pip install --user markupsafe==2.0.1` |
| OCR가 5초씩 느림 / `NvMapMem ... error 12` | `sudo jetson_clocks` (클럭 최대 고정), 메모리 확보 |
| 카메라 "찾을 수 없습니다" | `config.json` `camera_ip` 및 Pylon SDK 확인 |

> 핵심: **numpy 1.23.5 / pycuda 2022.2.2 / markupsafe 2.0.1** 세 버전 고정이 Jetson(Py3.8 + TRT8.5) 호환의 열쇠입니다.
