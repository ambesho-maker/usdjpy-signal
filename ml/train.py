"""LightGBM と XGBoost の比較（walk-forward検証 + コスト込みバックテスト）。
- 直近 TEST_MONTHS か月を、1か月ずつ「未来」としてテスト（学習は常にそれ以前のデータのみ）
- 学習データ末尾の一部で BUY/SELL のしきい値(pips)を決める（テスト期間は使わない）
- シグナル: 予測 > しきい値 → BUY / 予測 < -しきい値 → SELL / それ以外 → WAIT
- バックテスト: 1回に1ポジション、保有はホライズン分、往復コスト(スプレッド+スリッページ)を差し引く
※ 売買の自動発注は行わない。検証用のみ。
"""
import os
import sys
import numpy as np
import pandas as pd

from features import HORIZONS, build

TEST_MONTHS = int(os.environ.get("TEST_MONTHS", "6"))
COST_PIPS = float(os.environ.get("COST_PIPS", "0.5"))  # 往復: スプレッド+スリッページ（仮定）
MIN_TRADES = 30


def make_models():
    import lightgbm as lgb
    import xgboost as xgb
    return {
        "LightGBM": lambda: lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=5, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2),
    }


def backtest(times, pred, actual, thr, bars):
    """1ポジションずつ。戻り値: 各取引の [方向(+1/-1), 損益pips(コスト後)] の配列。"""
    pnls, free_at = [], None
    hold = pd.Timedelta(minutes=5 * bars)
    for t, p, a in zip(times, pred, actual):
        if free_at is not None and t < free_at:
            continue
        if p > thr:
            side = 1
        elif p < -thr:
            side = -1
        else:
            continue
        pnls.append((side, side * a - COST_PIPS))
        free_at = t + hold
    return np.array(pnls).reshape(-1, 2)


def choose_threshold(times, pred, actual, bars):
    cands = np.quantile(np.abs(pred), [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    best, best_pnl = np.inf, 0.0  # 見つからなければ WAIT のみ
    for thr in cands:
        pn = backtest(times, pred, actual, thr, bars)
        if len(pn) >= MIN_TRADES and pn[:, 1].sum() > best_pnl:
            best, best_pnl = thr, pn[:, 1].sum()
    return best


def max_drawdown(pnls):
    if len(pnls) == 0:
        return 0.0
    eq = np.cumsum(pnls)
    return float((np.maximum.accumulate(eq) - eq).max())


def month_starts(index, n):
    mx = index.max()
    last = pd.Timestamp(year=mx.year, month=mx.month, day=1, tz="UTC")
    return [last - pd.DateOffset(months=i) for i in range(n - 1, -1, -1)]


def run(models=None):
    models = models or make_models()
    df, X, y = build()
    ok = X.notna().all(axis=1)
    starts = month_starts(df.index, TEST_MONTHS)
    ends = starts[1:] + [df.index.max() + pd.Timedelta(minutes=5)]
    main_rows, diag_rows, month_rows = [], [], []
    for hname, bars in HORIZONS.items():
        col = f"ret_{hname}_pips"
        use = ok & y[col].notna()
        Xh, yh = X[use], y.loc[use, col]
        gap = pd.Timedelta(minutes=5 * bars)
        res = {m: {"p": [], "a": [], "bt": [], "label": []} for m in models}
        base_pnl = []
        for ms, me in zip(starts, ends):
            tr_all = Xh.index < ms - gap
            te = (Xh.index >= ms) & (Xh.index < me)
            if tr_all.sum() < 5000 or te.sum() == 0:
                continue
            cut = Xh.index[tr_all][int(tr_all.sum() * 0.8)]
            inner_tr = Xh.index < cut - gap
            inner_va = (Xh.index >= cut) & tr_all
            ytr = yh[inner_tr]
            lim = 4 * ytr.std()
            t_te, a_te = Xh.index[te], yh[te].values
            for name, make in models.items():
                m = make()
                m.fit(Xh[inner_tr], ytr.clip(-lim, lim))
                thr = choose_threshold(Xh.index[inner_va], m.predict(Xh[inner_va]),
                                       yh[inner_va].values, bars)
                pt = m.predict(Xh[te])
                res[name]["p"].append(pt)
                res[name]["a"].append(a_te)
                res[name]["bt"].append(backtest(t_te, pt, a_te, thr, bars))
                res[name]["label"].append(ms.strftime("%Y-%m"))
            base_pnl.append(backtest(t_te, np.ones(len(a_te)), a_te, 0.0, bars)[:, 1])
        if not base_pnl:
            continue
        bsum = float(np.concatenate(base_pnl).sum())
        for name, r in res.items():
            if not r["p"]:
                continue
            p, a = np.concatenate(r["p"]), np.concatenate(r["a"])
            bt = np.concatenate(r["bt"])
            pn, side = bt[:, 1], bt[:, 0]
            main_rows.append({
                "期間": hname, "モデル": name,
                "方向正解率%": round(float(((p > 0) == (a > 0)).mean() * 100), 1),
                "常に上昇の正解率%": round(float((a > 0).mean() * 100), 1),
                "相関": round(float(np.corrcoef(p, a)[0, 1]), 3),
                "取引数": len(pn),
                "勝率%": round(float((pn > 0).mean() * 100), 1) if len(pn) else 0.0,
                "合計pips": round(float(pn.sum()), 1),
                "平均pips": round(float(pn.mean()), 2) if len(pn) else 0.0,
                "最大DD pips": round(max_drawdown(pn), 1),
                "常にBUY合計pips": round(bsum, 1),
            })
            month_pnl = [float(b[:, 1].sum()) for b in r["bt"]]
            tval = (pn.mean() / (pn.std(ddof=1) / np.sqrt(len(pn)))) if len(pn) > 2 and pn.std() > 0 else 0.0
            diag_rows.append({
                "期間": hname, "モデル": name,
                "t値(2超が有意の目安)": round(float(tval), 2),
                "BUY数": int((side > 0).sum()), "SELL数": int((side < 0).sum()),
                "BUY損益": round(float(pn[side > 0].sum()), 1),
                "SELL損益": round(float(pn[side < 0].sum()), 1),
                "プラス月": f"{sum(v > 0 for v in month_pnl)}/{len(month_pnl)}",
            })
            mrow = {"期間": hname, "モデル": name}
            for lab, v in zip(r["label"], month_pnl):
                mrow[lab] = round(v, 1)
            month_rows.append(mrow)
    return pd.DataFrame(main_rows), pd.DataFrame(diag_rows), pd.DataFrame(month_rows)


def write_md(f, title, d):
    f.write(f"### {title}\n\n")
    f.write("| " + " | ".join(str(c) for c in d.columns) + " |\n")
    f.write("|" + "---|" * len(d.columns) + "\n")
    for _, r in d.iterrows():
        f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    f.write("\n")


if __name__ == "__main__":
    main_df, diag_df, month_df = run()
    print(f"設定: テスト{TEST_MONTHS}か月 / 往復コスト {COST_PIPS} pips")
    for title, d in (("結果", main_df), ("診断", diag_df), ("月別損益(pips)", month_df)):
        print(f"\n[{title}]")
        print(d.to_string(index=False))
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(f"## テスト{TEST_MONTHS}か月・往復コスト{COST_PIPS}pips\n\n")
            write_md(f, "結果", main_df)
            write_md(f, "診断", diag_df)
            write_md(f, "月別損益(pips)", month_df)
