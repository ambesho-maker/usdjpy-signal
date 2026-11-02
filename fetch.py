import csv, glob, json, os, sys, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone

OUTPUTSIZE = 96
FIELDS = ["datetime", "open", "high", "low", "close"]
FMT = "%Y-%m-%dT%H:%M:%SZ"


def fetch():
    q = urllib.parse.urlencode({
        "symbol": "USD/JPY", "interval": "5min", "outputsize": OUTPUTSIZE,
        "timezone": "UTC", "apikey": os.environ["TWELVEDATA_API_KEY"]})
    with urllib.request.urlopen(f"https://api.twelvedata.com/time_series?{q}", timeout=30) as r:
        d = json.load(r)
    if d.get("status") == "error" or "values" not in d:
        raise RuntimeError(f"Twelve Data error: {d.get('message', d)}")
    return d["values"]


def read_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["datetime"]: r for r in csv.DictReader(f)}


def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc)
    by_month, unfinished = {}, 0
    for v in fetch():
        start = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if start + timedelta(minutes=5) > now:
            unfinished += 1
            continue
        row = {"datetime": start.strftime(FMT), "open": v["open"], "high": v["high"],
               "low": v["low"], "close": v["close"]}
        by_month.setdefault(start.strftime("%Y%m"), []).append(row)
    saved = 0
    for month, rows in by_month.items():
        path = f"data/usdjpy_5min_{month}.csv"
        cur = read_csv(path)
        for r in rows:
            if r["datetime"] not in cur:
                cur[r["datetime"]] = r
                saved += 1
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for k in sorted(cur):
                w.writerow(cur[k])
    allrows = {}
    for p in glob.glob("data/usdjpy_5min_*.csv"):
        allrows.update(read_csv(p))
    latest = [allrows[k] for k in sorted(allrows, reverse=True)[:48]]
    with open("data/latest.json", "w") as f:
        json.dump({"updated": now.strftime(FMT), "signal": "WAIT",
                   "signal_note": "予測モデル未実装（検証完了まで投資判断に使用しないこと）",
                   "candles": latest}, f)
    print(f"OK saved={saved} unfinished_ignored={unfinished}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
