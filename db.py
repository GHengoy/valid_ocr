#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db.py — 일부인 검증 세션 저장소 (SQLite)

기존 date/<폴더>/list.json (플랫 JSON 파일, 쓰기 도중 정전 시 깨질 수 있음) 대신
SQLite 로 관리한다. SQLite 는 트랜잭션 단위 원자적 커밋 + WAL 모드라 쓰기 중
전원이 끊겨도 DB 가 깨지지 않는다.

- 세션 메타 + 2×14 검증 결과 그리드를 정규화 저장
- 이미지(PNG)는 기존처럼 date/<폴더>/ 에 그대로 두고 DB 엔 메타만 저장
- load_session() 은 기존 list.json 과 동일한 dict 형태를 반환하므로 app 로직은 그대로 동작

폴더 키는 app 의 state.folder_name 과 동일하게 "date/<폴더명>" 전체 경로를 쓴다.
"""

import os
import json
import time
import glob
import sqlite3
import datetime
import threading

MACHINES = 2
NUM_PER_MACHINE = 14

_DB_PATH = None
_lock = threading.Lock()


def _connect():
    con = sqlite3.connect(_DB_PATH, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")     # 원자적 쓰기 + 동시 읽기
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(db_path):
    """DB 파일 경로 설정 및 스키마 생성."""
    global _DB_PATH
    _DB_PATH = db_path
    con = _connect()
    try:
        with con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    folder          TEXT PRIMARY KEY,
                    ex_date_year    TEXT DEFAULT '',
                    ex_date_month   TEXT DEFAULT '',
                    ex_date_day     TEXT DEFAULT '',
                    ex_date         TEXT DEFAULT '',
                    jo_ya           TEXT DEFAULT '',
                    a_b             TEXT DEFAULT '',
                    machine         INTEGER DEFAULT 0,
                    sign_quality    INTEGER DEFAULT 0,
                    sign_production INTEGER DEFAULT 0,
                    updated_at      TEXT DEFAULT (datetime('now','localtime'))
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS cells (
                    folder    TEXT NOT NULL,
                    machine   INTEGER NOT NULL,
                    num       INTEGER NOT NULL,
                    ok        INTEGER DEFAULT 0,
                    date_str  TEXT DEFAULT '',
                    color     TEXT DEFAULT 'gray',
                    PRIMARY KEY (folder, machine, num),
                    FOREIGN KEY (folder) REFERENCES sessions(folder) ON DELETE CASCADE
                )
            """)
    finally:
        con.close()


def _empty_grids():
    return (
        [[0] * NUM_PER_MACHINE for _ in range(MACHINES)],
        [[""] * NUM_PER_MACHINE for _ in range(MACHINES)],
        [["gray"] * NUM_PER_MACHINE for _ in range(MACHINES)],
    )


def save_session(folder, data):
    """세션 저장(upsert). data 는 기존 list.json 과 동일한 dict 형태."""
    sign = data.get("sign_list", [0, 0]) or [0, 0]
    ok_list = data.get("ok_list") or _empty_grids()[0]
    date_list = data.get("date_list") or _empty_grids()[1]
    bc_list = data.get("b_c_list") or _empty_grids()[2]

    with _lock:
        con = _connect()
        try:
            with con:
                con.execute("""
                    INSERT INTO sessions
                        (folder,ex_date_year,ex_date_month,ex_date_day,ex_date,
                         jo_ya,a_b,machine,sign_quality,sign_production,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now','localtime'))
                    ON CONFLICT(folder) DO UPDATE SET
                        ex_date_year=excluded.ex_date_year,
                        ex_date_month=excluded.ex_date_month,
                        ex_date_day=excluded.ex_date_day,
                        ex_date=excluded.ex_date,
                        jo_ya=excluded.jo_ya,
                        a_b=excluded.a_b,
                        machine=excluded.machine,
                        sign_quality=excluded.sign_quality,
                        sign_production=excluded.sign_production,
                        updated_at=datetime('now','localtime')
                """, (
                    folder,
                    data.get("ex_date_year", ""), data.get("ex_date_month", ""),
                    data.get("ex_date_day", ""), data.get("ex_date", ""),
                    data.get("jo_ya", ""), data.get("a_b", ""),
                    int(data.get("machine", 0) or 0),
                    int(sign[0]) if len(sign) > 0 else 0,
                    int(sign[1]) if len(sign) > 1 else 0,
                ))
                rows = []
                for m in range(MACHINES):
                    for n in range(NUM_PER_MACHINE):
                        try:
                            ok_v = int(ok_list[m][n])
                        except (ValueError, TypeError, IndexError):
                            ok_v = 0
                        d_v = date_list[m][n] if m < len(date_list) and n < len(date_list[m]) else ""
                        c_v = bc_list[m][n] if m < len(bc_list) and n < len(bc_list[m]) else "gray"
                        rows.append((folder, m, n + 1, ok_v, d_v or "", c_v or "gray"))
                con.executemany("""
                    INSERT INTO cells (folder,machine,num,ok,date_str,color)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(folder,machine,num) DO UPDATE SET
                        ok=excluded.ok, date_str=excluded.date_str, color=excluded.color
                """, rows)
        finally:
            con.close()


def load_session(folder):
    """세션 로드 → 기존 list.json 과 동일한 dict (없으면 None)."""
    con = _connect()
    try:
        s = con.execute("SELECT * FROM sessions WHERE folder=?", (folder,)).fetchone()
        if s is None:
            return None
        ok_list, date_list, bc_list = _empty_grids()
        for c in con.execute("SELECT machine,num,ok,date_str,color FROM cells WHERE folder=?", (folder,)):
            m, n = c["machine"], c["num"] - 1
            if 0 <= m < MACHINES and 0 <= n < NUM_PER_MACHINE:
                ok_list[m][n] = c["ok"]
                date_list[m][n] = c["date_str"]
                bc_list[m][n] = c["color"]
        return {
            "ex_date_year": s["ex_date_year"],
            "ex_date_month": s["ex_date_month"],
            "ex_date_day": s["ex_date_day"],
            "ex_date": s["ex_date"],
            "jo_ya": s["jo_ya"],
            "a_b": s["a_b"],
            "machine": s["machine"],
            "ok_list": ok_list,
            "date_list": date_list,
            "b_c_list": bc_list,
            "sign_list": [s["sign_quality"], s["sign_production"]],
        }
    finally:
        con.close()


def set_sign(folder, index):
    """서명 상태 업데이트 (index 0=품질, 1=생산)."""
    col = "sign_quality" if index == 0 else "sign_production"
    with _lock:
        con = _connect()
        try:
            with con:
                con.execute(f"UPDATE sessions SET {col}=1, updated_at=datetime('now','localtime') WHERE folder=?", (folder,))
        finally:
            con.close()


def list_folders():
    """세션 폴더명(bare, 'date/' 제외) 최신순 리스트 — get_date_folders 호환."""
    con = _connect()
    try:
        rows = con.execute("SELECT folder FROM sessions").fetchall()
        names = []
        for r in rows:
            f = r["folder"]
            names.append(f[5:] if f.startswith("date/") else f)
        return sorted(names, reverse=True)
    finally:
        con.close()


def exists(folder):
    con = _connect()
    try:
        return con.execute("SELECT 1 FROM sessions WHERE folder=?", (folder,)).fetchone() is not None
    finally:
        con.close()


def delete_session(folder):
    with _lock:
        con = _connect()
        try:
            with con:
                con.execute("DELETE FROM sessions WHERE folder=?", (folder,))  # cells 는 CASCADE
        finally:
            con.close()


def delete_before(cutoff_date_str):
    """ex_date(YYYY-MM-DD) 가 cutoff 이전인 세션 폴더명 리스트 반환 후 DB 삭제.
    (실제 파일 폴더 삭제는 호출측에서 수행)"""
    con = _connect()
    try:
        rows = con.execute("SELECT folder, ex_date_year, ex_date_month, ex_date_day FROM sessions").fetchall()
    finally:
        con.close()
    to_delete = []
    for r in rows:
        y, m, d = r["ex_date_year"], r["ex_date_month"], r["ex_date_day"]
        if not (y and m and d):
            continue
        ymd = f"{y}-{m}-{d}"
        if ymd < cutoff_date_str:
            to_delete.append(r["folder"])
    for folder in to_delete:
        delete_session(folder)
    return to_delete


def backup_db(backup_dir, keep=14, min_interval_sec=0):
    """sessions.db 를 온라인 백업(WAL 안전)으로 backup_dir 에 복제한다.
    - 파일명: sessions_YYYYmmdd_HHMMSS.db
    - min_interval_sec 이내에 만든 백업이 이미 있으면 건너뜀(None 반환) — 잦은 재시작 시 중복 방지
    - keep 개수만 보관하고 오래된 백업 삭제
    생성한 백업 경로 반환(건너뛰면 None)."""
    if not _DB_PATH or not os.path.exists(_DB_PATH):
        return None
    os.makedirs(backup_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(backup_dir, "sessions_*.db")))
    if min_interval_sec and existing:
        age = time.time() - os.path.getmtime(existing[-1])
        if age < min_interval_sec:
            return None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(backup_dir, f"sessions_{ts}.db")
    src = _connect()
    try:
        dst = sqlite3.connect(dest)
        try:
            with dst:
                src.backup(dst)   # SQLite 온라인 백업 API (쓰기 중에도 일관성 보장)
        finally:
            dst.close()
    finally:
        src.close()

    # 회전: 최신 keep 개만 유지
    files = sorted(glob.glob(os.path.join(backup_dir, "sessions_*.db")))
    if keep and len(files) > keep:
        for old in files[:-keep]:
            try:
                os.remove(old)
            except OSError:
                pass
    return dest


def migrate_from_json(base_dir, data_dir_name="date"):
    """기존 date/<폴더>/list.json 중 '아직 DB에 없는 것만' 1회 이관한다.
    이미 DB에 있는 세션은 건너뛰므로, 최초 1회 이후에는 이관 건수 0(메시지 없음).
    원본 list.json 은 백업용으로 보존. 신규 이관 건수 반환."""
    data_dir = os.path.join(base_dir, data_dir_name)
    if not os.path.isdir(data_dir):
        return 0
    count = 0
    for name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, name)
        json_path = os.path.join(folder_path, "list.json")
        if not os.path.isdir(folder_path) or not os.path.exists(json_path):
            continue
        key = f"{data_dir_name}/{name}"
        if exists(key):
            continue  # 이미 DB에 있음 → 이관 불필요 (DB가 기준)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[db migrate] 건너뜀(JSON 손상): {name} — {e}")
            continue
        save_session(key, data)
        count += 1
    return count
