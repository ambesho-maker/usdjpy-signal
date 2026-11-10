"""実験: 外部データ(FRED)を足すと、翌日予測は良くなるか。
- 比較: 価格だけ vs 価格+外部データ（同じ手順・同じ設定・同じ行）
- 公表の遅れを見込み、判断日の2営業日前までの値だけを使う（未来情報を混ぜない）
- 古い8か月で比べ、最後の4か月は封印。開発期間で基準を満たした時だけ、REVEAL=true で1回だけ見る
- データはリポジトリに保存しない（実行のたびに取得）
"""
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from features import build
from train import COST_PIPS, month_starts

TEST_MONTHS = 12
HOLDOUT_MONTHS = 4
ENTRY_HOUR = 8       # UTC 08時台の足で判断（前向き検証と同じ）
BARS = 288
LAG_BDAYS = 2        # 公表の遅れの見込み
MIN_GAIN = 0.10      # 採用基準: 開発期間の相関が価格のみより0.10以上高い（約130日では相関の誤差が約0.09のため）
REVEAL = os.environ.get("REVEAL", "false").lower() == "true"
SERIES = {"us10y": "DGS10", "us2y": "DGS2", "sp500": "SP500", "vix": "VIXCLS",
          "wti": "DCOILWTICO", "usd": "DTWEXBGS", "nikkei": "NIKKEI225"}


def fetch_macro(start, end):
    key = os.environ["FRED_API_KEY"]
    start = (start - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    out = {}
    for name, sid in SERIES.items():
        q = urllib.parse.urlencode({"series_id": sid, "api_key": key, "file_type": "json",
                                    "observation_start": start})
        try:
            with urllib.request.urlopen(f"https://api.stlouisfed.org/fred/series/observations?{q}", timeout=30) as r:
                d = json.load(r)
            s = pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan)
                           for o in d.get("observations", [])})
            out[name] = s
            print(f"  {name}({sid}): {s.notna().sum()}日分")
        except Exception as e:  # キーが表示されないよう、種類だけ出す
            print(f"  {name}({sid}): 取得失敗 ({type(e).__name__})")
    if len(out) < 4:
        raise RuntimeError("外部データが4系列未満しか取れませんでした")
    return pd.DataFrame(out).sort_index()


def macro_features(raw):
    idx = pd.bdate_range(raw.index.min(), raw.index.max())
    d = raw.reindex(idx).ffill(limit=5)
    f = pd.DataFrame(index=idx)
    for c in ("us10y", "us2y"):
        if c in d:
            f[c], f[c + "_d1"], f[c + "_d5"] = d[c], d[c].diff(1), d[c].diff(5)
    if "us10y" in d and "us2y" in d:
        f["curve_10_2"] = d["us10y"] - d["us2y"]
    for c in ("sp500", "wti", "usd", "nikkei"):
        if c in d:
            f[c + "_r1"], f[c + "_r5"] = d[c] / d[c].shift(1) - 1, d[c] / d[c].shift(5) - 1
    if "vix" in d:
        f["vix"], f["vix_d1"], f["vix_r5"] = d["vix"], d["vix"].diff(1), d["vix"] / d["vix"].shift(5) - 1
    return f


def attach(feats, index):
    """5分足の各行に、判断日の LAG_BDAYS 営業日前までの外部データを割り当てる。"""
    days = index.tz_convert("UTC").normalize().tz_localize(None)
    uniq = days.unique()
    lag = pd.DatetimeIndex(np.busday_offset(uniq.values.astype("datetime64[D]"), -LAG_BDAYS, roll="backward"))
    mf = feats.reindex(lag)
    mf.index = uniq
    out = mf.reindex(days)
    out.index = index
    return out


def make_model():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=1500,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             importance_type="gain", n_jobs=2, verbose=-1)


def metrics(P, A):
    return {"判断日数": len(P), "相関": round(float(np.corrcoef(P, A)[0, 1]), 3),
            "方向一致%": round(float(((P > 0) == (A > 0)).mean() * 100), 1),
            "常に上昇%": round(float((A > 0).mean() * 100), 1),
            "ズレ平均pips": round(float(np.abs(A - P).mean()), 1),
            "符号売買の損益pips": round(float((np.sign(P) * A - COST_PIPS).sum()), 1)}


def run(factory=None, fetch=None):
    factory = factory or make_model
    df, X, y = build()
    raw = (fetch or fetch_macro)(df.index.min(), df.index.max())
    Xm = attach(macro_features(raw), X.index)
    recent = X.index > X.index.min() + pd.Timedelta(days=90)
    drop = [c for c in Xm.columns if Xm.loc[recent, c].isna().mean() > 0.05]
    Xm = Xm.drop(columns=drop)
    ok = X.notna().all(axis=1) & Xm.notna().all(axis=1) & y["ret_1d_pips"].notna()
    Xb, Xa, yh = X[ok], pd.concat([X, Xm], axis=1)[ok], y.loc[ok, "ret_1d_pips"]
    entry = (Xb.index.hour == ENTRY_HOUR) & (Xb.index.minute == 0) & (Xb.index.day_of_week <= 3)
    starts = month_starts(df.index, TEST_MONTHS)
    ends = starts[1:] + [df.index.max() + pd.Timedelta(minutes=5)]
    gap = pd.Timedelta(minutes=5 * BARS)
    res = {"dev": {"base": [], "macro": [], "act": []}, "hold": {"base": [], "macro": [], "act": []}}
    imp = []
    for i, (ms, me) in enumerate(zip(starts, ends)):
        per = "dev" if i < TEST_MONTHS - HOLDOUT_MONTHS else "hold"
        tr, te = Xb.index < ms - gap, entry & (Xb.index >= ms) & (Xb.index < me)
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        ytr = yh[tr]
        lim = 4 * ytr.std()
        mb, ma = factory(), factory()
        mb.fit(Xb[tr], ytr.clip(-lim, lim))
        ma.fit(Xa[tr], ytr.clip(-lim, lim))
        res[per]["base"].append(mb.predict(Xb[te]))
        res[per]["macro"].append(ma.predict(Xa[te]))
        res[per]["act"].append(yh[te].values)
        if per == "dev" and hasattr(ma, "feature_importances_"):
            imp.append(pd.Series(ma.feature_importances_ / max(ma.feature_importances_.sum(), 1e-9), index=Xa.columns))
    info = {"外部データ系列": ", ".join(raw.columns), "除外した特徴量(欠損多)": ", ".join(drop) or "なし",
            "学習に使える行数": int(ok.sum())}
    rows = []

    def add(label, per):
        if not res[per]["act"]:
            return None
        A = np.concatenate(res[per]["act"])
        m = {}
        for key, name in (("base", "価格のみ"), ("macro", "価格+外部データ")):
            m[key] = metrics(np.concatenate(res[per][key]), A)
            rows.append({"期間": label, "モデル": name, **m[key]})
        return m

    dev = add("開発(古い8か月)", "dev")
    gain = dev["macro"]["相関"] - dev["base"]["相関"] if dev else None
    passed = gain is not None and gain >= MIN_GAIN
    verdict = [f"開発期間の相関の差: {gain:+.3f}（基準 +{MIN_GAIN:.2f} 以上）→ {'達成' if passed else '未達'}"]
    if passed and REVEAL:
        hold = add("最後の4か月(封印解除)", "hold")
        ok2 = hold["macro"]["相関"] > 0 and hold["macro"]["相関"] >= hold["base"]["相関"]
        verdict.append(f"最後の4か月: 相関 {hold['macro']['相関']:+.3f}（価格のみ {hold['base']['相関']:+.3f}）→ "
                       f"{'採用候補' if ok2 else '不採用'}")
    elif passed:
        verdict.append("最後の4か月は封印中（見る場合は REVEAL=true で1回だけ実行）")
    else:
        verdict.append("開発期間で基準未達のため、最後の4か月は見ません（外部データは不採用）")
    top = (pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False).head(8).round(3)
           if imp else pd.Series(dtype=float))
    return pd.DataFrame(rows), verdict, top, info


def write_md(f, title, d):
    f.write(f"### {title}\n\n| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n")
    for _, r in d.iterrows():
        f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    f.write("\n")


if __name__ == "__main__":
    table, verdict, top, info = run()
    print(table.to_string(index=False))
    print("\n".join(verdict))
    print("重要度(上位):\n" + top.to_string())
    print(info)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write("## 外部データの実験（毎日1回の判断・翌日予測）\n\n")
            write_md(f, "結果", table)
            f.write("### 判定\n\n" + "\n".join(f"- {v}" for v in verdict) + "\n\n")
            write_md(f, "価格+外部データの重要度（上位）", top.rename("重要度").reset_index().rename(columns={"index": "特徴量"}))
            f.write("\n".join(f"- {k}: {v}" for k, v in info.items()) + "\n")
