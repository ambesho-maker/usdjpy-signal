"""前向き検証と同じ条件のバックテスト: 「毎日1回・月〜木・決まった時刻」に翌日予測を判断する。
- 学習・しきい値の決め方は ml/forward.py と同じ（月ごとに再学習する点だけ簡略化）
- 比較する4通り: A 現行(しきい値を最適化) / B しきい値なし(常に売買) / C 固定しきい値(上位30%) / D 常にBUY
"""
import os
import numpy as np
import pandas as pd

from features import build
from train import COST_PIPS, choose_threshold, make_models, max_drawdown, month_starts

TEST_MONTHS = int(os.environ.get("TEST_MONTHS", "12"))
ENTRY_HOUR = int(os.environ.get("ENTRY_HOUR", "8"))   # UTC。08時台の足の終値で判断
BARS = 288


def stats(name, side, actual, pred):
    take = side != 0
    pn = side[take] * actual[take] - COST_PIPS
    n = len(pn)
    t = float(pn.mean() / (pn.std(ddof=1) / np.sqrt(n))) if n > 2 and pn.std() > 0 else 0.0
    return {
        "方式": name, "判断日数": len(side), "取引数": n,
        "WAIT率%": round(float((~take).mean() * 100), 1),
        "BUY": int((side > 0).sum()), "SELL": int((side < 0).sum()),
        "勝率%": round(float((pn > 0).mean() * 100), 1) if n else 0.0,
        "合計pips": round(float(pn.sum()), 1), "平均pips": round(float(pn.mean()), 2) if n else 0.0,
        "最大DD": round(max_drawdown(pn), 1), "t値": round(t, 2),
    }


def run(factory=None):
    factory = factory or make_models()["LightGBM"]
    df, X, y = build()
    ok = X.notna().all(axis=1)
    use = ok & y["ret_1d_pips"].notna()
    Xh, yh = X[use], y.loc[use, "ret_1d_pips"]
    entry = (Xh.index.hour == ENTRY_HOUR) & (Xh.index.minute == 0) & (Xh.index.day_of_week <= 3)
    starts = month_starts(df.index, TEST_MONTHS)
    ends = starts[1:] + [df.index.max() + pd.Timedelta(minutes=5)]
    gap = pd.Timedelta(minutes=5 * BARS)
    P, A, thrA, thrC, months, per_month = [], [], [], [], [], []
    for ms, me in zip(starts, ends):
        tr_all = Xh.index < ms - gap
        te = entry & (Xh.index >= ms) & (Xh.index < me)
        if tr_all.sum() < 5000 or te.sum() == 0:
            continue
        cut = Xh.index[tr_all][int(tr_all.sum() * 0.8)]
        inner_tr, inner_va = Xh.index < cut - gap, (Xh.index >= cut) & tr_all
        ytr = yh[inner_tr]
        lim = 4 * ytr.std()
        m = factory()
        m.fit(Xh[inner_tr], ytr.clip(-lim, lim))
        pv = m.predict(Xh[inner_va])
        ta = choose_threshold(Xh.index[inner_va], pv, yh[inner_va].values, BARS)
        tc = float(np.quantile(np.abs(pv), 0.7))
        pt = m.predict(Xh[te])
        P.append(pt); A.append(yh[te].values)
        thrA += [ta] * len(pt); thrC += [tc] * len(pt); months += [ms.strftime("%Y-%m")] * len(pt)
    P, A = np.concatenate(P), np.concatenate(A)
    thrA, thrC, months = np.array(thrA), np.array(thrC), np.array(months)
    sideA = np.where(P > thrA, 1, np.where(P < -thrA, -1, 0))
    sideB = np.sign(P).astype(int)
    sideC = np.where(P > thrC, 1, np.where(P < -thrC, -1, 0))
    sideD = np.ones(len(P), dtype=int)
    main = pd.DataFrame([stats("A 現行(しきい値を最適化)", sideA, A, P), stats("B しきい値なし", sideB, A, P),
                         stats("C 固定しきい値(上位30%)", sideC, A, P), stats("D 常にBUY", sideD, A, P)])
    acc = pd.DataFrame([{
        "判断日数": len(P), "ズレ平均pips": round(float(np.abs(A - P).mean()), 1),
        "方向一致%": round(float(((P > 0) == (A > 0)).mean() * 100), 1),
        "常に上昇%": round(float((A > 0).mean() * 100), 1), "相関": round(float(np.corrcoef(P, A)[0, 1]), 3)}])
    rows = []
    for mo in dict.fromkeys(months):
        k = months == mo
        take = sideA[k] != 0
        rows.append({"月": mo, "判断日数": int(k.sum()), "しきい値A": round(float(thrA[k][0]), 1) if np.isfinite(thrA[k][0]) else "inf",
                     "A取引数": int(take.sum()), "A損益pips": round(float((sideA[k][take] * A[k][take] - COST_PIPS).sum()), 1),
                     "B損益pips": round(float((sideB[k] * A[k] - COST_PIPS).sum()), 1)})
    return main, acc, pd.DataFrame(rows)


def write_md(f, title, d):
    f.write(f"### {title}\n\n| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n")
    for _, r in d.iterrows():
        f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    f.write("\n")


if __name__ == "__main__":
    main_df, acc_df, month_df = run()
    print(f"設定: 直近{TEST_MONTHS}か月 / 判断時刻 UTC {ENTRY_HOUR}時台 月〜木 / 往復コスト {COST_PIPS}pips")
    for title, d in (("方式の比較", main_df), ("予測の精度", acc_df), ("月別(方式A・B)", month_df)):
        print(f"\n[{title}]\n{d.to_string(index=False)}")
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(f"## 毎日1回の判断でのバックテスト（直近{TEST_MONTHS}か月・コスト{COST_PIPS}pips）\n\n")
            write_md(f, "方式の比較", main_df)
            write_md(f, "予測の精度", acc_df)
            write_md(f, "月別(方式A・B)", month_df)
