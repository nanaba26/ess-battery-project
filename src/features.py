"""
features.py
- Batch 1 / 2 / 3 의 .mat 파일에서 셀(배터리) 단위 피처 표를 만든다.
- 결과 : results/features.csv  (셀 1개 = 1행)

실행 : 프로젝트 폴더에서  python src/features.py
"""
import os
import re
import gc
import warnings
import logging

import numpy as np
import pandas as pd
import mat73

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)   # mat73 의 'string not supported' 로그 숨김

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(ROOT, "results", "features.csv")

BATCH_FILES = {
    "Batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


# ---------------------------------------------------------------------------
# 1. 로딩 : mat73 의 dict-of-lists → list-of-dicts (셀 1개 = dict 1개)
# ---------------------------------------------------------------------------
def load_batch(fname):
    mat = mat73.loadmat(os.path.join(DATA_DIR, fname))
    b = mat["batch"]
    if isinstance(b, dict):
        keys = list(b.keys())
        n = len(b[keys[0]])
        b = [{k: b[k][j] for k in keys} for j in range(n)]
    return b


# ---------------------------------------------------------------------------
# 2. 정제 : 0값(위장 결측치)과 범위 밖 값은 NaN 처리
# ---------------------------------------------------------------------------
def clean(arr, low, high):
    a = np.asarray(arr, dtype=float).copy()
    a[(a <= low) | (a >= high)] = np.nan
    return a


def parse_policy(p):
    """'5.4C(40%)-3.6C' → (5.4, 40, 3.6). 형식이 다르면 NaN."""
    m = re.match(r"([\d.]+)C\((\d+)%\)-([\d.]+)C", str(p))
    return (float(m[1]), float(m[2]), float(m[3])) if m else (np.nan, np.nan, np.nan)


# ---------------------------------------------------------------------------
# 3. 셀 1개 → 피처 1행
#    summary 배열 index 0 = cycle 1 (0값) → cycle n 은 index n-1
#    Qdlin 은 cycles 리스트의 index 10, 100 사용 (원논문 Q100 - Q10)
# ---------------------------------------------------------------------------
def cell_features(cell):
    f = {}

    # --- Target ---
    try:
        f["cycle_life"] = float(np.asarray(cell["cycle_life"]).squeeze())
    except Exception:
        f["cycle_life"] = np.nan

    # --- 충전 조건 ---
    policy = cell.get("policy_readable") or cell.get("policy") or "unknown"
    f["policy"] = str(policy)
    f["C1"], f["Q1_pct"], f["C2"] = parse_policy(policy)

    s = cell["summary"]
    qd = clean(s["QDischarge"], 0.8, 1.3)     # 정상 용량 범위만 (공칭 약 1.08 Ah)
    ir = clean(s["IR"], 0.0, 0.1)
    tavg = clean(s["Tavg"], 0.0, 60.0)
    ct = clean(s["chargetime"], 0.0, 60.0)

    # --- [Q3] ΔQ(V) = Q100(V) - Q10(V) ---
    qdlin = cell["cycles"]["Qdlin"]
    f["log_dq_var"] = f["log_dq_min"] = f["dq_mean"] = np.nan
    if len(qdlin) > 100 and qdlin[10] is not None and qdlin[100] is not None:
        dq = np.asarray(qdlin[100], float) - np.asarray(qdlin[10], float)
        f["log_dq_var"] = np.log10(np.var(dq))
        f["log_dq_min"] = np.log10(np.abs(np.min(dq)))
        f["dq_mean"] = np.mean(dq)

    # --- [Q2] 용량 : cycle 2 용량, 초기 최대-cycle2, cycle 91~100 기울기 ---
    early = qd[1:100]                                  # cycle 2 ~ 100
    f["qd_cycle2"] = qd[1] if len(qd) > 1 else np.nan
    f["qd_max_minus_c2"] = np.nanmax(early) - f["qd_cycle2"] if np.isfinite(early).any() else np.nan
    seg = qd[90:100]                                   # cycle 91 ~ 100
    ok = np.isfinite(seg)
    f["qd_slope_91_100"] = np.polyfit(np.arange(91, 101)[ok], seg[ok], 1)[0] if ok.sum() >= 3 else np.nan

    # --- [Q5] 온도·충전시간·저항 (cycle 2 ~ 100, 0값 제외) ---
    f["mean_Tavg"] = np.nanmean(tavg[1:100])
    f["med_chargetime_2_6"] = np.nanmedian(ct[1:6])    # 초기 5사이클 충전시간
    f["ir_min"] = np.nanmin(ir[1:100])
    f["ir_100_minus_2"] = ir[99] - ir[1] if len(ir) > 99 else np.nan

    # --- [Q2] EOL 도달 여부 : 마지막 유효 용량이 80%(약 0.88 Ah) 근처까지 내려갔는가 ---
    valid = qd[np.isfinite(qd)]
    f["qd_last"] = valid[-1] if len(valid) else np.nan
    f["eol_reached"] = bool(len(valid) and valid[-1] <= 0.90)
    return f


# ---------------------------------------------------------------------------
# 4. 실행 : 배치를 하나씩 불러와 피처만 남기고 메모리에서 삭제
# ---------------------------------------------------------------------------
def main():
    rows = []
    for name, fname in BATCH_FILES.items():
        print(f"▶ {name} 로딩 중... ({fname})")
        batch = load_batch(fname)
        for i, cell in enumerate(batch):
            f = cell_features(cell)
            f.update(batch=name, cell_id=i)
            rows.append(f)
        print(f"  셀 {len(batch)}개 처리 완료")
        del batch
        gc.collect()

    df = pd.DataFrame(rows)
    front = ["batch", "cell_id", "cycle_life", "eol_reached", "policy"]
    df = df[front + [c for c in df.columns if c not in front]]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n저장 완료 → {OUT_PATH}  shape={df.shape}")
    print(df.groupby("batch").agg(셀수=("cell_id", "count"),
                                 수명없음=("cycle_life", lambda x: x.isna().sum()),
                                 EOL미도달=("eol_reached", lambda x: (~x).sum()),
                                 ΔQ없음=("log_dq_var", lambda x: x.isna().sum())))


if __name__ == "__main__":
    main()
