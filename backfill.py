import csv, json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone

from fetch import FIELDS, FMT, read_csv

WINDOW_DAYS = 14   # 1回の取得範囲（5000本の上限に収まる大きさ）
PAUSE = 10         # 無料枠の毎分上限(8)を超えないための待ち秒数


def fetch_window(start, end):
    q = urllib.parse.urlencode({
        "symbol": "USD/JPY", "interval": "5min", "outputsize": 5000,
        "timezone": "UTC", "order": "asc",
        "start_date": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "apikey": os.environ["TWELVEDATA_API_KEY"]})
    with urllib.request.urlopen(f"https://api.twelvedata.com/time_series?{q}", timeout=60) as r:
        d = json.load(r)
    if d.get("status") == "error":
        msg = str(d.get("message", ""))
        if "no data" in msg.lower():
            return []
        raise RuntimeError(f"Twelve Data error: {msg}")
    return d.get("values", [])


def main():
    days = min(max(int(os.environ.get("DAYS", "90")), 1), 730)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    cur = now - timedelta(days=days)
    by_month, total, calls = {}, 0, 0
    while cur < now:
        w_end = min(cur + timedelta(days=WINDOW_DAYS), now)
        if calls:
            time.sleep(PAUSE)
        values = fetch_window(cur, w_end)
        calls += 1
        print(f"window {cur:%Y-%m-%d} -> {w_end:%Y-%m-%d}: {len(values)} rows", flush=True)
        for v in values:
            s = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if s + timedelta(minutes=5) > now:
                continue
            row = {"datetime": s.strftime(FMT), "open": v["open"], "high": v["high"],
                   "low": v["low"], "close": v["close"]}
            by_month.setdefault(s.strftime("%Y%m"), {})[row["datetime"]] = row
        cur = w_end
    os.makedirs("data", exist_ok=True)
    saved = 0
    for month, rows in by_month.items():
        path = f"data/usdjpy_5min_{month}.csv"
        existing = read_csv(path)
        for k, r in rows.items():
            if k not in existing:
                existing[k] = r
                saved += 1
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for k in sorted(existing):
                w.writerow(existing[k])
    print(f"OK api_calls={calls} new_rows={saved}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
