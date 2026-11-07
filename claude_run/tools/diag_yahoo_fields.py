# Diagnostic: what Yahoo (yfinance) returns for news, analyst targets and
# valuation on the GitHub runner, so best_picks.py can read the right fields.
import json, yfinance as yf

print("yfinance", yf.__version__)
for t in ["MPC", "NEM", "TEAM", "HNRG"]:
    tk = yf.Ticker(t)
    print(f"\n==================== {t}")
    try:
        info = tk.info or {}
        print("info keys:", len(info))
        for k in ["shortName", "longName", "targetMeanPrice", "numberOfAnalystOpinions", "forwardPE",
                  "forwardEps", "trailingPegRatio", "pegRatio", "recommendationMean", "currentPrice"]:
            print(f"  {k}: {info.get(k)!r}")
    except Exception as e:
        print("info error:", type(e).__name__, e)
    try:
        print("analyst_price_targets:", tk.analyst_price_targets)
    except Exception as e:
        print("analyst_price_targets error:", type(e).__name__, e)
    try:
        news = tk.news or []
        print("news items:", len(news))
        for it in news[:3]:
            c = it.get("content", it)
            fin = c.get("finance") if isinstance(c, dict) else None
            print("  keys:", sorted(c.keys())[:25] if isinstance(c, dict) else type(c))
            print("  title:", c.get("title"))
            print("  finance:", json.dumps(fin)[:300] if fin else None)
            for k in ("relatedTickers", "stockTickers", "tickers"):
                if k in c or k in it:
                    print(f"  {k}:", c.get(k) or it.get(k))
    except Exception as e:
        print("news error:", type(e).__name__, e)
