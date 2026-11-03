"""前向き検証（ペーパートレード）。発注は行わない。記録と答え合わせのみ。
毎日: ①過去の記録の答え合わせ ②LightGBM翌日予測で今日のシグナルを記録 ③成績を集計
手順は ml/train.py で検証した方法と同じ（学習の80%で学習、残り20%でしきい値を決定）。
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from features import PIP, build
from train import COST_PIPS, choose_threshold, make_models

FMT = "%Y-%m-%dT%H:%M:%SZ"
BARS = 288                     # 24時間 = 288本
OUT = "forward"
COLS = ["entry_time", "entry_close", "pred_pips", "threshold", "signal", "status",
        "exit_time", "exit_close", "gross_pips", "net_pips"]
PASS_TRADES = 60               # 本判定に必要な取引回数
PASS_MEAN_PIPS = 4.3           # バックテスト平均(8.6pips, コスト1.0)の半分


def load_log():
    path = f"{OUT}/predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"exit_time": str, "exit_close": str}).astype(object)
    return pd.DataFrame(columns=COLS).astype(object)


def settle(log, df):
    """24時間後の足で答え合わせ。見つからなければ待つ。30分以上過ぎても無ければ欠損。"""
    log = log.astype(object)  # 文字列型のままだと数値を書き込めない
    latest = df.index.max()
    for i, r in log[log["status"] == "open"].iterrows():
        target = pd.Timestamp(r["entry_time"]) + pd.Timedelta(hours=24)
        cand = df.index[(df.index >= target) & (df.index <= target + pd.Timedelta(minutes=30))]
        if len(cand):
            t = cand[0]
            gross = (df.loc[t, "close"] - float(r["entry_close"])) * PIP
            side = {"BUY": 1, "SELL": -1}.get(r["signal"], 0)
            log.loc[i, ["status", "exit_time", "exit_close", "gross_pips", "net_pips"]] = [
                "settled", t.strftime(FMT), round(float(df.loc[t, "close"]), 5),
                round(float(gross), 2), round(float(side * gross - COST_PIPS if side else 0.0), 2)]
        elif latest > target + pd.Timedelta(minutes=30):
            log.loc[i, "status"] = "missing"
    return log


def record_today(log, df, X, y, now, force, factory):
    latest = df.index.max()
    if not force and now.weekday() > 3:
        return log, "月〜木以外のため記録しません"
    if (now - latest) > pd.Timedelta(hours=3, minutes=5):
        return log, f"最新データが古い({latest})ため記録しません"
    et = latest.strftime(FMT)
    if et in set(log["entry_time"]):
        return log, "この足は記録済み"
    if not force and (log["entry_time"].astype(str).str[:10] == et[:10]).any():
        return log, "今日は記録済み"
    ok = X.notna().all(axis=1)
    if not ok.loc[latest]:
        return log, "特徴量が揃っていません"
    use = ok & y["ret_1d_pips"].notna()
    Xh, yh = X[use], y.loc[use, "ret_1d_pips"]
    if len(Xh) < 5000:
        return log, "学習データ不足"
    gap = pd.Timedelta(minutes=5 * BARS)
    cut = Xh.index[int(len(Xh) * 0.8)]
    inner_tr, inner_va = Xh.index < cut - gap, Xh.index >= cut
    lim = 4 * yh[inner_tr].std()
    m = factory()
    m.fit(Xh[inner_tr], yh[inner_tr].clip(-lim, lim))
    thr = choose_threshold(Xh.index[inner_va], m.predict(Xh[inner_va]), yh[inner_va].values, BARS)
    pred = float(m.predict(X.loc[[latest]])[0])
    sig = "BUY" if pred > thr else "SELL" if pred < -thr else "WAIT"
    log.loc[len(log)] = [et, round(float(df.loc[latest, "close"]), 5), round(pred, 2),
                         round(float(thr), 3) if np.isfinite(thr) else "inf", sig, "open",
                         "", "", "", ""]
    log = log.astype(object)
    return log, f"記録: {et} {sig} (予測 {pred:+.1f}pips)"


def summarize(log):
    s = log[(log["status"] == "settled") & (log["signal"] != "WAIT")]
    pn = s["net_pips"].astype(float).values
    n = len(pn)
    eq = np.cumsum(pn)
    t = float(pn.mean() / (pn.std(ddof=1) / np.sqrt(n))) if n > 2 and pn.std() > 0 else 0.0
    out = {
        "updated": datetime.now(timezone.utc).strftime(FMT),
        "cost_pips": COST_PIPS, "records": int(len(log)),
        "wait": int((log["signal"] == "WAIT").sum()), "open": int((log["status"] == "open").sum()),
        "trades": n, "wins": int((pn > 0).sum()),
        "total_pips": round(float(pn.sum()), 1) if n else 0.0,
        "mean_pips": round(float(pn.mean()), 2) if n else 0.0,
        "max_dd_pips": round(float((np.maximum.accumulate(eq) - eq).max()), 1) if n else 0.0,
        "t_value": round(t, 2),
        "criteria": f"取引{PASS_TRADES}回以上で、合計>0 かつ 平均>={PASS_MEAN_PIPS}pips",
    }
    if n < PASS_TRADES:
        out["verdict"] = f"判定前（あと{PASS_TRADES - n}回）"
    else:
        out["verdict"] = "合格" if (out["total_pips"] > 0 and out["mean_pips"] >= PASS_MEAN_PIPS) else "不合格"
    return out


def main(now=None, force=False, factory=None, out_dir=None):
    global OUT
    if out_dir:
        OUT = out_dir
    os.makedirs(OUT, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    factory = factory or make_models()["LightGBM"]
    df, X, y = build()
    log = settle(load_log(), df)
    log, msg = record_today(log, df, X, y, pd.Timestamp(now), force, factory)
    log = log.sort_values("entry_time")
    log.to_csv(f"{OUT}/predictions.csv", index=False)
    summ = summarize(log)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(msg)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(f"## 前向き検証（ペーパートレード）\n\n{msg}\n\n")
            f.write(f"- 判定: **{summ['verdict']}**（基準: {summ['criteria']}）\n")
            f.write(f"- 取引 {summ['trades']}回 / 勝ち {summ['wins']} / 合計 {summ['total_pips']}pips / "
                    f"平均 {summ['mean_pips']}pips / 最大DD {summ['max_dd_pips']}pips / t値 {summ['t_value']}\n")
            f.write(f"- WAIT {summ['wait']}回 / 答え合わせ待ち {summ['open']}件 / 往復コスト {summ['cost_pips']}pips\n\n")
            f.write("| 時刻(UTC) | シグナル | 予測pips | 結果pips | 状態 |\n|---|---|---|---|---|\n")
            for _, r in log.tail(10).iloc[::-1].iterrows():
                f.write(f"| {r['entry_time']} | {r['signal']} | {r['pred_pips']} | {r['net_pips']} | {r['status']} |\n")


if __name__ == "__main__":
    try:
        main(force=os.environ.get("FORCE", "false").lower() == "true")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
