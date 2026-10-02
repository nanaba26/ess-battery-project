"""
error_analysis.py
- 최종 모델(Linear + log var ΔQ)이 Batch 2 에서 크게 틀린 원인 분석
- 입력 : results/features.csv, results/predictions.csv  (train.py 실행 후)
- 출력 : results/error_analysis.csv, results/figures/error_analysis.png

※ 이 파일의 모든 분석은 모델 확정 '이후'의 사후 분석이며, 모델 선택에는 사용하지 않음
실행 : 프로젝트 폴더에서  python src/error_analysis.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_data, split_holdout, mape, RES, FIG   # 같은 데이터·분할 재사용

warnings.filterwarnings("ignore")
COLORS = {"Batch1": "tab:blue", "Batch2": "tab:orange", "Batch3": "tab:green"}


def main():
    b1, b2, b3 = load_data(exclude_unreliable=True)
    tr, va = split_holdout(b1)
    model = LinearRegression().fit(tr[["log_dq_var"]], tr["log_life"])   # 최종 모델과 동일
    out = []

    # ---------------------------------------------------------------
    # 1. 오차 방향 : 과대예측인가 과소예측인가 (부호 있는 오차)
    # ---------------------------------------------------------------
    print("[1] 배치별 부호 있는 오차 (예측 - 실제) / 실제  (+ = 수명을 길게 예측)")
    for name, d in [("Valid(B1)", va), ("Test(B2)", b2), ("Test(B3)", b3)]:
        e = (10 ** model.predict(d[["log_dq_var"]]) - d["cycle_life"]) / d["cycle_life"] * 100
        print(f"  {name:10s} 평균 {e.mean():+6.1f}%  중앙값 {e.median():+6.1f}%  과대예측 비율 {(e > 0).mean()*100:5.1f}%")
        out.append(["1_signed_error", name, round(e.mean(), 1), round(e.median(), 1), round((e > 0).mean() * 100, 1)])

    # ---------------------------------------------------------------
    # 2. 배치별 직선 : 기울기는 같은데 절편만 다른가?
    # ---------------------------------------------------------------
    print("\n[2] 배치별 log10(life) = a * log10(var ΔQ) + b")
    lines = {}
    for name, d in [("Batch1", b1), ("Batch2", b2), ("Batch3", b3)]:
        a, b = np.polyfit(d["log_dq_var"], d["log_life"], 1)
        lines[name] = (a, b)
        print(f"  {name} : 기울기 a = {a:+.3f}, 절편 b = {b:.3f}  (n={len(d)})")
        out.append(["2_batch_line", name, round(a, 3), round(b, 3), len(d)])
    shift = 10 ** (lines["Batch2"][1] - lines["Batch1"][1] + (lines["Batch2"][0] - lines["Batch1"][0]) * b2["log_dq_var"].mean())
    print(f"  → 같은 ΔQ 수준에서 Batch2 수명은 Batch1 대비 약 {shift*100:.0f}% 수준")

    # ---------------------------------------------------------------
    # 3. 같은 충전 방식, 다른 배치 : 4.8C(80%)-4.8C
    # ---------------------------------------------------------------
    print("\n[3] 같은 충전 방식(4.8C(80%)-4.8C)의 배치 간 비교")
    allb = pd.concat([b1, b2, b3])
    allb["base_policy"] = allb["policy"].str.replace("-newstructure", "", regex=False)
    same = allb[allb["base_policy"] == "4.8C(80%)-4.8C"].groupby("batch").agg(
        n=("cell_id", "count"), mean_life=("cycle_life", "mean"), mean_log_dq_var=("log_dq_var", "mean"))
    print(same.round(2).to_string())
    for bt, r in same.iterrows():
        out.append(["3_same_policy_4.8C", bt, round(r["mean_life"], 1), round(r["mean_log_dq_var"], 2), int(r["n"])])

    # ---------------------------------------------------------------
    # 4. Batch 2 오차가 충전 조건·온도와 관련 있는가?
    # ---------------------------------------------------------------
    print("\n[4] Batch 2 부호 있는 오차 vs 셀 특성 (Spearman)")
    e2 = (10 ** model.predict(b2[["log_dq_var"]]) - b2["cycle_life"]) / b2["cycle_life"] * 100
    for c in ["C1", "C2", "Q1_pct", "med_chargetime_2_6", "mean_Tavg", "cycle_life"]:
        rho, p = spearmanr(b2[c], e2, nan_policy="omit")
        print(f"  {c:20s} ρ = {rho:+.2f}  (p = {p:.3f})")
        out.append(["4_b2_error_corr", c, round(rho, 2), round(p, 3), ""])
    ns = b2["policy"].str.contains("newstructure")
    print(f"  newstructure 셀 평균 오차 {e2[ns].mean():+.1f}% (n={ns.sum()}) / 그 외 {e2[~ns].mean():+.1f}% (n={(~ns).sum()})")

    # ---------------------------------------------------------------
    # 5. 개선 실험 : Batch 2 셀 k개로 절편만 보정 (기울기는 그대로)
    #    → 실제 운영에서 '새 로트의 기준 셀 몇 개'를 먼저 시험하는 상황을 가정
    # ---------------------------------------------------------------
    print("\n[5] 절편 보정 실험 : Batch 2 셀 k개로 오프셋만 보정 → 나머지 셀 평가 (무작위 200회 평균)")
    base = mape(b2["cycle_life"].values, 10 ** model.predict(b2[["log_dq_var"]]))
    print(f"  보정 없음      : MAPE {base:.2f}%")
    rng = np.random.RandomState(0)
    for k in [3, 5, 8]:
        sc = []
        for _ in range(200):
            idx = rng.choice(len(b2), k, replace=False)
            cal, rest = b2.iloc[idx], b2.drop(b2.index[idx])
            off = np.mean(cal["log_life"] - model.predict(cal[["log_dq_var"]]))
            sc.append(mape(rest["cycle_life"].values, 10 ** (model.predict(rest[["log_dq_var"]]) + off)))
        print(f"  기준 셀 {k}개 보정 : MAPE {np.mean(sc):.2f}% (± {np.std(sc):.2f})")
        out.append(["5_recalibration", f"k={k}", round(np.mean(sc), 2), round(np.std(sc), 2), base])

    pd.DataFrame(out, columns=["analysis", "item", "v1", "v2", "v3"]).to_csv(
        os.path.join(RES, "error_analysis.csv"), index=False)

    # ---------------------------------------------------------------
    # 그림 : (좌) 배치별 직선  (우) 실제 수명별 오차
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    xs = np.linspace(-5.3, -2.9, 50)
    for name, d in [("Batch1", b1), ("Batch2", b2), ("Batch3", b3)]:
        a, b = lines[name]
        axes[0].scatter(d["log_dq_var"], d["log_life"], s=25, c=COLORS[name], alpha=0.75,
                        edgecolors="white", label=f"{name} (a={a:.2f}, b={b:.2f})")
        axes[0].plot(xs, a * xs + b, c=COLORS[name], lw=1.5)
    axes[0].plot(xs, model.predict(pd.DataFrame({"log_dq_var": xs})), "k--", lw=1.2, label="Final model (B1 train)")
    axes[0].set_xlabel("log10(var ΔQ)"); axes[0].set_ylabel("log10(cycle life)")
    axes[0].set_title("Same slope, shifted intercept (Batch 2)"); axes[0].legend(fontsize=9)

    for name, d in [("Valid(B1)", va), ("Batch2", b2), ("Batch3", b3)]:
        e = (10 ** model.predict(d[["log_dq_var"]]) - d["cycle_life"]) / d["cycle_life"] * 100
        axes[1].scatter(d["cycle_life"], e, s=25, alpha=0.8, edgecolors="white",
                        c=COLORS.get(name, "tab:blue"), label=name)
    axes[1].axhline(0, c="k", lw=1)
    axes[1].set_xlabel("Actual cycle life"); axes[1].set_ylabel("Signed error (%)  (+ = over-predict)")
    axes[1].set_title("Prediction error by actual life"); axes[1].legend()
    plt.tight_layout()
    path = os.path.join(FIG, "error_analysis.png")
    plt.savefig(path, dpi=130)
    print(f"\n그림 저장 → {path}")


if __name__ == "__main__":
    main()
