"""
train.py
- 입력 : results/features.csv  (src/features.py 결과, 셀 1개 = 1행)
- 학습 : Batch 1  /  테스트 : Batch 2  (추가 검증 : Batch 3)
- Target : log10(cycle_life)  → 예측 후 10^x 로 복원하여 MAPE 계산

데이터 분할 (가이드 기준)
- Valid : Batch 1 Hold-out 20%, '충전 방식' 단위로 묶어서 분리 (같은 방식 셀이 train/valid 로 갈라지는 누수 방지)
- Train : 나머지 Batch 1 에서 GroupKFold(5) 교차검증 평균
- Test  : Batch 2, 모델 확정 후 1회

실행 : 프로젝트 폴더에서  python src/train.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.base import clone

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

SEED = 42
TARGET_MAPE = 9.1          # 원논문 Regression 성능 (Severson et al., 2019)

# ---------------------------------------------------------------------------
# 1. 피처 세트 (DAY 1 전략)
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    # 원논문 'variance model' 과 같은 단일 핵심 피처
    "F0_dq_var": ["log_dq_var"],
    # A : ΔQ(V) 곡선 요약 + 초기 용량 변화 (배치 간 안정적인 셀 고유 신호)
    "FA_dq_capacity": ["log_dq_var", "log_dq_min", "dq_mean",
                       "qd_cycle2", "qd_max_minus_c2", "qd_slope_91_100"],
    # B : A + 충전 조건/온도 (Batch 1 안에서만 상관이 강했던 신호)
    "FB_plus_charge": ["log_dq_var", "log_dq_min", "dq_mean",
                       "qd_cycle2", "qd_max_minus_c2", "qd_slope_91_100",
                       "med_chargetime_2_6", "mean_Tavg", "C1"],
}

# ---------------------------------------------------------------------------
# 2. 후보 모델 (1순위 : 선형 + 정규화 / 2순위 : 트리 계열 비교)
#    하이퍼파라미터 후보는 Train CV 로 선택
# ---------------------------------------------------------------------------
def pipe(model):
    # 결측 대체 · 스케일링도 파이프라인 안에 두어 train 에서만 fit (데이터 누수 방지)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

MODELS = {
    "Linear":     [pipe(LinearRegression())],
    "Ridge":      [pipe(Ridge(alpha=a)) for a in [0.01, 0.1, 1, 10]],
    "Lasso":      [pipe(Lasso(alpha=a, max_iter=50000)) for a in [0.0005, 0.001, 0.005, 0.01]],
    "ElasticNet": [pipe(ElasticNet(alpha=a, l1_ratio=r, max_iter=50000))
                   for a in [0.001, 0.005, 0.01] for r in [0.2, 0.5, 0.8]],
    "RandomForest": [pipe(RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_leaf=2,
                                                random_state=SEED))
                     for n in [300] for d in [3, 5, None]],
    "GradientBoosting": [pipe(GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=2,
                                                        random_state=SEED))
                         for n in [100, 300] for lr in [0.05, 0.1]],
}
LINEAR = {"Linear", "Ridge", "Lasso", "ElasticNet"}


def mape(y_true_life, y_pred_life):
    return float(np.mean(np.abs((y_true_life - y_pred_life) / y_true_life)) * 100)


def predict_life(model, X):
    return 10 ** model.predict(X)


# ---------------------------------------------------------------------------
# 3. 데이터 준비
# ---------------------------------------------------------------------------
def load_data(exclude_unreliable=True):
    df = pd.read_csv(os.path.join(RES, "features.csv"))
    df = df[df["cycle_life"].notna()].copy()                # 수명 미기록(다른 실험) 제외
    df["log_life"] = np.log10(df["cycle_life"])
    b1 = df[df["batch"] == "Batch1"]
    if exclude_unreliable:                                   # EOL 미도달 → 수명 라벨 신뢰 불가
        b1 = b1[b1["eol_reached"]]
    return b1.reset_index(drop=True), df[df["batch"] == "Batch2"], df[df["batch"] == "Batch3"]


def split_holdout(b1):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(gss.split(b1, groups=b1["policy"]))
    return b1.iloc[tr_idx].reset_index(drop=True), b1.iloc[va_idx].reset_index(drop=True)


def cv_mape(model, tr, feats, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for a, b in gkf.split(tr, groups=tr["policy"]):
        m = clone(model).fit(tr.iloc[a][feats], tr.iloc[a]["log_life"])
        scores.append(mape(tr.iloc[b]["cycle_life"].values, predict_life(m, tr.iloc[b][feats])))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 4. 모델 탐색 : (피처 세트 × 모델) 마다 CV 로 하이퍼파라미터 선택 → Valid 평가
# ---------------------------------------------------------------------------
def search(tr, va, te, b3):
    rows, fitted = [], {}
    for fs, feats in FEATURE_SETS.items():
        for name, cands in MODELS.items():
            cvs = [cv_mape(c, tr, feats) for c in cands]
            best = clone(cands[int(np.argmin(cvs))]).fit(tr[feats], tr["log_life"])
            fitted[(fs, name)] = best
            rows.append({
                "feature_set": fs, "model": name,
                "params": str(best.steps[-1][1].get_params(deep=False))
                          if name != "Linear" else "-",
                "Train_CV_MAPE": round(min(cvs), 2),
                "Valid_MAPE": round(mape(va["cycle_life"].values, predict_life(best, va[feats])), 2),
                # 아래 두 열은 '사후 분석'용 (모델 선택에는 사용하지 않음)
                "Test_B2_MAPE(posthoc)": round(mape(te["cycle_life"].values, predict_life(best, te[feats])), 2),
                "Test_B3_MAPE(posthoc)": round(mape(b3["cycle_life"].values, predict_life(best, b3[feats])), 2),
            })
    return pd.DataFrame(rows), fitted


def report_table(train_cv, valid, test, b3=None):
    rows = [
        ["Train (Batch 1 CV)", train_cv, ""],
        ["Valid (Batch 1 Hold-out)", valid, ""],
        ["Test (Batch 2)", test, ""],
        ["Gap (Train-Valid)", valid - train_cv, "(+) : 과적합 의심"],
        ["Gap (Valid-Test)", test - valid, "(+) : 배치간 일반화 저하 의심"],
        ["Gap (Target-Test)", test - TARGET_MAPE, f"Target : 원논문 {TARGET_MAPE}%"],
    ]
    if b3 is not None:
        rows += [["Test (Batch 3)", b3, "추가 검증"],
                 ["Gap (Batch2-Batch3)", b3 - test, "Test 성능 간 비교"],
                 ["Gap (Target-Test, Batch 3)", b3 - TARGET_MAPE, "Batch 3 기준, 원논문 성능 비교"]]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"]).round({"MAPE (%)": 2})


def main():
    b1, b2, b3 = load_data(exclude_unreliable=True)
    tr, va = split_holdout(b1)
    print(f"[분할] Batch1 학습 가능 셀 {len(b1)}개 → Train {len(tr)} / Valid {len(va)}  |  "
          f"Test(B2) {len(b2)}  |  B3 {len(b3)}")
    print(f"       Valid 충전 방식 : {sorted(va['policy'].unique())}")
    assert not set(tr["policy"]) & set(va["policy"]), "train/valid 충전 방식 중복 (누수)"

    # ---- 4-1. 모델 탐색 (선택 기준 = Valid MAPE) ----
    comp, fitted = search(tr, va, b2, b3)
    comp = comp.sort_values("Valid_MAPE").reset_index(drop=True)
    comp.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    print("\n[모델 비교] (선택 기준 : Valid_MAPE, Test 열은 사후 분석용)")
    print(comp.drop(columns="params").to_string(index=False))

    best = comp.iloc[0]
    fs, name = best["feature_set"], best["model"]
    feats, model = FEATURE_SETS[fs], fitted[(fs, name)]
    print(f"\n[최종 모델] {name} + {fs}  ({best['params']})")

    # ---- 4-2. 최종 모델 성능 리포트 (가이드 Format) ----
    test_mape = mape(b2["cycle_life"].values, predict_life(model, b2[feats]))
    b3_mape = mape(b3["cycle_life"].values, predict_life(model, b3[feats]))
    perf = report_table(best["Train_CV_MAPE"], best["Valid_MAPE"], test_mape, b3_mape)
    perf.to_csv(os.path.join(RES, "model_performance.csv"), index=False)
    print("\n[성능 리포트]")
    print(perf.to_string(index=False))

    # ---- 4-3. 셀별 예측 저장 (오류 분석용) ----
    pred = []
    for nm, d in [("Valid(B1)", va), ("Test(B2)", b2), ("Test(B3)", b3)]:
        p = predict_life(model, d[feats])
        pred.append(pd.DataFrame({"set": nm, "batch": d["batch"].values, "cell_id": d["cell_id"].values,
                                  "policy": d["policy"].values, "cycle_life": d["cycle_life"].values,
                                  "pred_life": p.round(0),
                                  "APE(%)": (np.abs(p - d["cycle_life"].values) / d["cycle_life"].values * 100).round(2),
                                  "log_dq_var": d["log_dq_var"].values}))
    pred = pd.concat(pred, ignore_index=True)
    pred.to_csv(os.path.join(RES, "predictions.csv"), index=False)
    print("\n[Batch 2 오차 상위 5개 셀]")
    print(pred[pred["set"] == "Test(B2)"].sort_values("APE(%)", ascending=False).head(5).to_string(index=False))

    # ---- 4-4. 실험 1 : 신뢰 불가 셀(EOL 미도달 10개) 포함 시 ----
    b1_all, _, _ = load_data(exclude_unreliable=False)
    tr_all = pd.concat([b1_all[~b1_all["policy"].isin(va["policy"])]], ignore_index=True)
    m_all = clone(model).fit(tr_all[feats], tr_all["log_life"])
    exp1 = pd.DataFrame([
        ["제외 (메인)", len(tr), best["Valid_MAPE"], round(test_mape, 2)],
        ["포함", len(tr_all), round(mape(va["cycle_life"].values, predict_life(m_all, va[feats])), 2),
         round(mape(b2["cycle_life"].values, predict_life(m_all, b2[feats])), 2)],
    ], columns=["EOL 미도달 셀", "Train 셀 수", "Valid MAPE", "Test(B2) MAPE"])
    exp1.to_csv(os.path.join(RES, "exp1_unreliable_cells.csv"), index=False)
    print("\n[실험 1] EOL 미도달 셀 포함/제외 (같은 Valid 셀로 비교)")
    print(exp1.to_string(index=False))

    # ---- 4-5. 실험 2 : 선형 vs 트리 외삽 (같은 피처 세트, 사후 분석) ----
    exp2 = comp[comp["feature_set"] == fs][["model", "Valid_MAPE", "Test_B2_MAPE(posthoc)"]]
    tree = fitted[(fs, "RandomForest")]
    print("\n[실험 2] 같은 피처 세트에서 모델별 성능 (Test 는 사후 분석)")
    print(exp2.to_string(index=False))
    print(f"  학습셋 수명 범위 : {tr['cycle_life'].min():.0f} ~ {tr['cycle_life'].max():.0f}")
    print(f"  RandomForest 의 Batch 2 예측 최솟값 : {predict_life(tree, b2[feats]).min():.0f}"
          f"  / 최종 모델 예측 최솟값 : {predict_life(model, b2[feats]).min():.0f}"
          f"  / 실제 최솟값 : {b2['cycle_life'].min():.0f}")

    # ---- 5. 그림 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lim = [300, 2100]
    for nm, d, c in [("Train(B1)", tr, "gray"), ("Valid(B1)", va, "tab:blue"),
                     ("Test(B2)", b2, "tab:orange"), ("Test(B3)", b3, "tab:green")]:
        axes[0].scatter(d["cycle_life"], predict_life(model, d[feats]), s=30, c=c, label=nm,
                        edgecolors="white", alpha=0.85)
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Actual cycle life"); axes[0].set_ylabel("Predicted cycle life")
    axes[0].set_title(f"Final model : {name} ({fs})"); axes[0].legend()

    for mdl, c, lab in [(model, "tab:blue", f"{name} (final)"), (tree, "tab:red", "RandomForest")]:
        axes[1].scatter(b2["cycle_life"], predict_life(mdl, b2[feats]), s=30, c=c, label=lab,
                        edgecolors="white", alpha=0.85)
    axes[1].axvspan(tr["cycle_life"].min(), tr["cycle_life"].max(), color="gray", alpha=0.12,
                    label="Train life range")
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xlim([300, 1300]); axes[1].set_ylim([300, 1300])
    axes[1].set_xlabel("Actual cycle life (Batch 2)"); axes[1].set_ylabel("Predicted")
    axes[1].set_title("Extrapolation : linear vs tree (Batch 2)"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pred_vs_actual.png"), dpi=130)
    print(f"\n그림 저장 → {os.path.join(FIG, 'pred_vs_actual.png')}")


if __name__ == "__main__":
    main()
