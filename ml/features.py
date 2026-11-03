"""特徴量と正解ラベルの作成。python ml/features.py で品質チェックを表示する。
- 特徴量は「その足が確定した時点で分かる情報」だけを使う（未来情報を混ぜない）
- ラベルは 1時間後 / 4時間後 / 24時間後 の終値の変化（pips）
"""
import glob
import numpy as np
import pandas as pd

HORIZONS = {"1h": 12, "4h": 48, "1d": 288}  # 5分足の本数
PIP = 100.0  # USD/JPY は 1円 = 100 pips


def load_data(pattern="data/usdjpy_5min_*.csv"):
    frames = [pd.read_csv(p) for p in sorted(glob.glob(pattern))]
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df[["open", "high", "low", "close"]].astype(float)


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    f = pd.DataFrame(index=df.index)
    r1 = np.log(c).diff()
    for n in (1, 3, 12, 48):
        f[f"ret_{n}"] = np.log(c).diff(n)
    f["rsi_14"] = _rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = macd * PIP, sig * PIP, (macd - sig) * PIP
    for n in (20, 50, 200):
        f[f"sma_{n}_gap"] = c / c.rolling(n).mean() - 1
    f["sma_20_50_gap"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - mid) / (2 * sd).replace(0, np.nan)
    f["bb_width"] = 4 * sd / mid
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14_pips"] = tr.ewm(alpha=1 / 14, adjust=False).mean() * PIP
    f["body_pips"] = (c - o) * PIP
    f["upper_wick_pips"] = (h - np.maximum(o, c)) * PIP
    f["lower_wick_pips"] = (np.minimum(o, c) - l) * PIP
    f["range_pips"] = (h - l) * PIP
    f["vol_12"], f["vol_48"] = r1.rolling(12).std(), r1.rolling(48).std()
    hour = f.index.hour + f.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow"] = f.index.day_of_week
    return f


def make_labels(df):
    """各時刻のちょうど N 分後の足が存在する場合のみラベルを付ける（週末をまたぐ分は欠損）。"""
    y = pd.DataFrame(index=df.index)
    for name, bars in HORIZONS.items():
        future = df["close"].reindex(df.index + pd.Timedelta(minutes=5 * bars))
        y[f"ret_{name}_pips"] = (future.values - df["close"].values) * PIP
    return y


def build():
    df = load_data()
    X = make_features(df)
    y = make_labels(df)
    return df, X, y


if __name__ == "__main__":
    df, X, y = build()
    print(f"期間: {df.index.min()} ～ {df.index.max()}")
    print(f"行数: {len(df)} / 特徴量: {X.shape[1]}個")
    dt = df.index.to_series().diff().dropna()
    gaps = dt[(dt > pd.Timedelta(minutes=5)) & (dt < pd.Timedelta(hours=40))]
    print(f"週末以外の欠損(5分超の間隔): {len(gaps)}回 / 最大 {dt[dt < pd.Timedelta(hours=40)].max()}")
    if len(gaps):
        print("大きい欠損TOP3:")
        for t, g in gaps.sort_values(ascending=False).head(3).items():
            print(f"  {t} の直前 {g}")
    ok = X.notna().all(axis=1)
    print(f"特徴量が揃った行: {int(ok.sum())} (先頭の約200本は助走期間)")
    for name in HORIZONS:
        col = y[f"ret_{name}_pips"]
        use = ok & col.notna()
        s = col[use]
        print(f"[{name}] 学習可能 {int(use.sum())}行 | 平均 {s.mean():+.2f}pips 標準偏差 {s.std():.1f}pips 上昇割合 {(s > 0).mean():.1%}")
    flat = (df["high"] == df["low"]).sum()
    print(f"土日の行: {int((df.index.day_of_week >= 5).sum())} / 値動きゼロの足: {int(flat)}")
    for d, x in df.groupby(df.index.day_of_week):
        print(f"曜日{d}(0=月): {len(x)}行 平均レンジ {((x['high'] - x['low']) * 100).mean():.2f}pips 時間帯 {x.index.hour.nunique()}/24")
