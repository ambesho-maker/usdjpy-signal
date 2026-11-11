"""実験: 分類モデル（±BAND pips以上の動きの確率）で売買判定すると、回帰より良くなるか。
- 比較対象は現行の回帰（ml/daily_backtest.py の方式A）を、同じ判断日で再計算したもの
- ラベルは3クラス: UP(>=+BAND) / FLAT / DOWN(<=-BAND)。特徴量は features.py の24個をそのまま使う
- 判断は前向き検証と同じ条件: UTC 08時の足・月〜木のみ・往復コストは環境変数 COST_PIPS（ワークフローでは1.0）
- 古い8か月（開発）で判定し、最後の4か月は封印。開発で基準を全て満たした時だけ REVEAL=true で1回だけ見る
- 対象ホライズンは本命の 4h と、対照の 1d（環境変数 HORIZONS で変更可、例: HORIZONS=4h）
※ 売買の自動発注は行わない。検証用のみ。前向き検証（forward.py）には組み込まない。
"""
import os

import numpy as np
import pandas as pd

from daily_backtest import stats
from features import HORIZONS, build
from train import COST_PIPS, choose_threshold, month_starts

TEST_MONTHS = int(os.environ.get("TEST_MONTHS", "12"))
HOLDOUT_MONTHS = int(os.environ.get("HOLDOUT_MONTHS", "4"))
ENTRY_HOUR = int(os.environ.get("ENTRY_HOUR", "8"))   # UTC。08時台の足の終値で判断
BAND = float(os.environ.get("BAND_PIPS", "15"))       # 「大きく動いた」とみなす幅(pips)
REVEAL = os.environ.get("REVEAL", "false").lower() == "true"
TARGETS = [s.strip() for s in os.environ.get("HORIZONS", "4h,1d").split(",") if s.strip() in HORIZONS]

CLS_DOWN, CLS_FLAT, CLS_UP = 0, 1, 2
PROB_CANDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]     # 方式Eで探す確率しきい値の候補
DIR_PROB_MIN = 0.40                                    # 方式E: 方向の確率自体の下限
MIN_TRADES_THR = 30                                    # しきい値決定に必要な最小取引数（train.py と同じ）
MIN_DEV_TRADES = int(os.environ.get("MIN_DEV_TRADES", "40"))  # 採用基準3: 開発期間の方式Eの取引数
AUC_MIN = 0.52                                         # 採用基準4


def make_label(ret_pips, band=BAND):
    """実績pipsから3クラスラベルを作る（UP=2 / FLAT=1 / DOWN=0）。"""
    return pd.Series(np.where(ret_pips >= band, CLS_UP,
                              np.where(ret_pips <= -band, CLS_DOWN, CLS_FLAT)),
                     index=ret_pips.index, name="label")


def make_model():
    import lightgbm as lgb
    # train.py の回帰とハイパーパラメータを揃える（差が「分類にしたこと」だけになるように）
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        importance_type="gain", n_jobs=2, verbose=-1)


def make_regressor():
    """ベースライン（方式A）用の回帰モデル。train.py の LightGBM と同一パラメータ。
    train.make_models() は XGBoost も import するため、依存を増やさないようここで定義する。"""
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1)


def proba_columns(model, X):
    """predict_proba を [DOWN, FLAT, UP] の固定順で返す。"""
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, j]
    return out


def side_from_proba(proba, thr, gated=True):
    """確率から売買方向(+1/-1/0)を作る。thr はスカラーでも月ごとの配列でも可。
    gated=True（方式E/GのうちE）: 方向の確率が DIR_PROB_MIN 以上で、FLAT より大きい時だけ売買する。"""
    down, flat, up = proba[:, 0], proba[:, 1], proba[:, 2]
    score = up - down
    if gated:
        buy = (score > thr) & (up >= DIR_PROB_MIN) & (up > flat)
        sell = (score < -thr) & (down >= DIR_PROB_MIN) & (down > flat)
    else:
        buy, sell = score > thr, score < -thr
    return np.where(buy, 1, np.where(sell, -1, 0))


def pnl_nonoverlap(times, side, actual, bars):
    """1ポジション制約つきの損益（コスト後）。train.backtest() と同じ進み方。"""
    pnls, free_at = [], None
    hold = pd.Timedelta(minutes=5 * bars)
    for t, s, a in zip(times, side, actual):
        if s == 0 or (free_at is not None and t < free_at):
            continue
        pnls.append(s * a - COST_PIPS)
        free_at = t + hold
    return np.array(pnls)


def choose_prob_threshold(times, proba, actual, bars):
    """choose_threshold() の分類版。inner validation で合計損益が最大の確率しきい値を選ぶ。
    取引数 MIN_TRADES_THR 以上でプラスになる候補が無ければ inf（取引なし）。"""
    best, best_pnl = np.inf, 0.0
    for thr in PROB_CANDS:
        pn = pnl_nonoverlap(times, side_from_proba(proba, thr), actual, bars)
        if len(pn) >= MIN_TRADES_THR and pn.sum() > best_pnl:
            best, best_pnl = thr, float(pn.sum())
    return best


def prob_metrics(proba, label):
    """確率の質: AUC・logloss・Brier（基準率モデルとの比較）と、P(UP) の校正表。"""
    from sklearn.metrics import log_loss, roc_auc_score
    down, up = proba[:, 0], proba[:, 2]
    is_up, is_down = (label == CLS_UP).astype(float), (label == CLS_DOWN).astype(float)

    def auc(prob, truth):
        try:
            return float(roc_auc_score(truth, prob))
        except ValueError:
            return float("nan")

    brier = float(np.mean((up - is_up) ** 2) + np.mean((down - is_down) ** 2)) / 2
    base = float(np.mean((is_up.mean() - is_up) ** 2) + np.mean((is_down.mean() - is_down) ** 2)) / 2
    m = {"判断数": len(label), "AUC_UP": round(auc(up, is_up), 3), "AUC_DOWN": round(auc(down, is_down), 3),
         "logloss": round(float(log_loss(label, proba, labels=[0, 1, 2])), 3),
         "Brier": round(brier, 4), "Brier基準率": round(base, 4)}
    bins = pd.cut(up, bins=np.arange(0, 1.01, 0.1), include_lowest=True)
    rel = pd.DataFrame({"bin": bins, "is_up": is_up}).groupby("bin", observed=True)["is_up"].agg(["count", "mean"])
    pred_mean = pd.Series(up).groupby(bins, observed=True).mean()
    rel_df = pd.DataFrame({"P(UP)の範囲": [str(i) for i in rel.index], "件数": rel["count"].astype(int).values,
                           "予測平均": np.round(pred_mean.values, 3), "実績UP率": np.round(rel["mean"].values, 3)})
    return m, rel_df


def total_pnl(side, actual):
    take = side != 0
    return float((side[take] * actual[take] - COST_PIPS).sum()) if take.any() else 0.0


def run_horizon(hname, df, X, y):
    bars = HORIZONS[hname]
    col = f"ret_{hname}_pips"
    ok = X.notna().all(axis=1)
    use = ok & y[col].notna()
    Xh, yh = X[use], y.loc[use, col]
    yc = make_label(yh)
    entry = (Xh.index.hour == ENTRY_HOUR) & (Xh.index.minute == 0) & (Xh.index.day_of_week <= 3)
    starts = month_starts(df.index, TEST_MONTHS)
    ends = starts[1:] + [df.index.max() + pd.Timedelta(minutes=5)]
    gap = pd.Timedelta(minutes=5 * bars)
    dev_n = TEST_MONTHS - HOLDOUT_MONTHS if TEST_MONTHS > HOLDOUT_MONTHS else TEST_MONTHS
    res = {p: {"proba": [], "label": [], "act": [], "reg": [], "thrA": [], "thrE": [], "thrG": [], "month": []}
           for p in ("dev", "hold")}
    imp = []
    reg_factory = make_regressor
    for i, (ms, me) in enumerate(zip(starts, ends)):
        per = "dev" if i < dev_n else "hold"
        tr_all = Xh.index < ms - gap
        te = entry & (Xh.index >= ms) & (Xh.index < me)
        if tr_all.sum() < 5000 or te.sum() == 0:
            continue
        cut = Xh.index[tr_all][int(tr_all.sum() * 0.8)]
        inner_tr, inner_va = Xh.index < cut - gap, (Xh.index >= cut) & tr_all
        # 分類モデル: inner_tr で学習、inner_va で確率しきい値を決める
        clf = make_model()
        clf.fit(Xh[inner_tr], yc[inner_tr])
        pv = proba_columns(clf, Xh[inner_va])
        thr_e = choose_prob_threshold(Xh.index[inner_va], pv, yh[inner_va].values, bars)
        thr_g = float(np.quantile(np.abs(pv[:, 2] - pv[:, 0]), 0.7))
        # ベースラインの回帰（方式A）: daily_backtest.py と同じ手順
        ytr = yh[inner_tr]
        lim = 4 * ytr.std()
        reg = reg_factory()
        reg.fit(Xh[inner_tr], ytr.clip(-lim, lim))
        thr_a = choose_threshold(Xh.index[inner_va], reg.predict(Xh[inner_va]), yh[inner_va].values, bars)
        r = res[per]
        r["proba"].append(proba_columns(clf, Xh[te]))
        r["label"].append(yc[te].values)
        r["act"].append(yh[te].values)
        r["reg"].append(reg.predict(Xh[te]))
        n = int(te.sum())
        r["thrA"] += [thr_a] * n
        r["thrE"] += [thr_e] * n
        r["thrG"] += [thr_g] * n
        r["month"] += [ms.strftime("%Y-%m")] * n
        if per == "dev" and hasattr(clf, "feature_importances_"):
            imp.append(pd.Series(clf.feature_importances_ / max(clf.feature_importances_.sum(), 1e-9),
                                 index=Xh.columns))

    def aggregate(per):
        r = res[per]
        if not r["act"]:
            return None
        proba = np.concatenate(r["proba"])
        A = np.concatenate(r["act"])
        score = proba[:, 2] - proba[:, 0]
        sides = {
            "E 確率しきい値(最適化)": side_from_proba(proba, np.array(r["thrE"])),
            "F 確率0.50固定": np.where(proba[:, 2] >= 0.5, 1, np.where(proba[:, 0] >= 0.5, -1, 0)),
            "G 上位30%(score分位点)": side_from_proba(proba, np.array(r["thrG"]), gated=False),
            "A 回帰(現行)": np.where(np.concatenate(r["reg"]) > np.array(r["thrA"]), 1,
                                     np.where(np.concatenate(r["reg"]) < -np.array(r["thrA"]), -1, 0)),
            "D 常にBUY": np.ones(len(A), dtype=int),
        }
        main = pd.DataFrame([stats(k, s, A, score) for k, s in sides.items()])
        pm, rel = prob_metrics(proba, np.concatenate(r["label"]))
        months = np.array(r["month"])
        mrows = []
        for mo in dict.fromkeys(months):
            k = months == mo
            mrows.append({"月": mo, "判断日数": int(k.sum()),
                          "E損益pips": round(total_pnl(sides["E 確率しきい値(最適化)"][k], A[k]), 1),
                          "A損益pips": round(total_pnl(sides["A 回帰(現行)"][k], A[k]), 1)})
        return {"main": main, "prob": pd.DataFrame([pm]), "rel": rel,
                "month": pd.DataFrame(mrows), "sides": sides, "act": A, "pm_raw": pm}

    dev = aggregate("dev")
    if dev is None:
        return None
    e_row = dev["main"].iloc[0]
    tot_e, tot_a = total_pnl(dev["sides"]["E 確率しきい値(最適化)"], dev["act"]), \
        total_pnl(dev["sides"]["A 回帰(現行)"], dev["act"])
    auc_ok = min(dev["pm_raw"]["AUC_UP"], dev["pm_raw"]["AUC_DOWN"]) >= AUC_MIN
    brier_ok = dev["pm_raw"]["Brier"] < dev["pm_raw"]["Brier基準率"]
    checks = [
        (f"方式Eの合計損益がプラス（{tot_e:+.1f} pips）", tot_e > 0),
        (f"方式Eが方式A（回帰）を上回る（A: {tot_a:+.1f} pips）", tot_e > tot_a),
        (f"方式Eの取引数が{MIN_DEV_TRADES}以上（{int(e_row['取引数'])}回）", int(e_row["取引数"]) >= MIN_DEV_TRADES),
        (f"AUC（UP/DOWNとも{AUC_MIN}以上: {dev['pm_raw']['AUC_UP']:.3f} / {dev['pm_raw']['AUC_DOWN']:.3f}）"
         f" かつ Brierが基準率より良い（{dev['pm_raw']['Brier']:.4f} < {dev['pm_raw']['Brier基準率']:.4f}）",
         auc_ok and brier_ok),
    ]
    passed = all(ok for _, ok in checks)
    verdict = [f"- {t}: {'達成' if ok else '未達'}" for t, ok in checks]
    hold = None
    if passed and REVEAL:
        hold = aggregate("hold")
        if hold is not None:
            ht_e = total_pnl(hold["sides"]["E 確率しきい値(最適化)"], hold["act"])
            ht_a = total_pnl(hold["sides"]["A 回帰(現行)"], hold["act"])
            adopt = ht_e > 0 and ht_e >= ht_a
            verdict.append(f"- 最後の{HOLDOUT_MONTHS}か月（封印解除）: 方式E {ht_e:+.1f} pips"
                           f"（方式A {ht_a:+.1f} pips）→ {'採用候補' if adopt else '不採用'}")
    elif passed:
        verdict.append(f"- 開発期間で基準達成。最後の{HOLDOUT_MONTHS}か月は封印中"
                       "（見る場合は REVEAL=true で1回だけ実行）")
    else:
        verdict.append(f"- 開発期間で基準未達のため、最後の{HOLDOUT_MONTHS}か月は見ません（分類モデルは不採用）")
    top = (pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False).head(8).round(3)
           if imp else pd.Series(dtype=float))
    return {"dev": dev, "hold": hold, "verdict": verdict, "top": top}


def write_md(f, title, d):
    f.write(f"### {title}\n\n| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n")
    for _, r in d.iterrows():
        f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    f.write("\n")


def run():
    df, X, y = build()
    out = {}
    for hname in TARGETS:
        r = run_horizon(hname, df, X, y)
        if r is not None:
            out[hname] = r
    return out


if __name__ == "__main__":
    results = run()
    print(f"設定: 直近{TEST_MONTHS}か月（開発{TEST_MONTHS - HOLDOUT_MONTHS if TEST_MONTHS > HOLDOUT_MONTHS else TEST_MONTHS}"
          f"＋封印{HOLDOUT_MONTHS}）/ 判断 UTC {ENTRY_HOUR}時 月〜木 / バンド ±{BAND:.0f} pips / 往復コスト {COST_PIPS} pips")
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    f = open(path, "a") if path else None
    for hname, r in results.items():
        print(f"\n===== {hname} =====")
        for title, d in (("方式の比較（開発）", r["dev"]["main"]), ("確率の質（開発）", r["dev"]["prob"]),
                         ("月別（開発）", r["dev"]["month"])):
            print(f"\n[{title}]\n{d.to_string(index=False)}")
        print("\n[判定]\n" + "\n".join(r["verdict"]))
        print("\n重要度（上位）:\n" + r["top"].to_string())
        if f:
            f.write(f"## 分類モデルの実験（{hname} / ±{BAND:.0f} pips）\n\n")
            write_md(f, "方式の比較（開発）", r["dev"]["main"])
            write_md(f, "確率の質（開発）", r["dev"]["prob"])
            write_md(f, "校正 P(UP)（開発）", r["dev"]["rel"])
            write_md(f, "月別（開発）", r["dev"]["month"])
            if r["hold"] is not None:
                write_md(f, "方式の比較（封印期間）", r["hold"]["main"])
            f.write("### 判定\n\n" + "\n".join(r["verdict"]) + "\n\n")
            write_md(f, "特徴量の重要度（上位）",
                     r["top"].rename("重要度").reset_index().rename(columns={"index": "特徴量"}))
    if f:
        f.close()
