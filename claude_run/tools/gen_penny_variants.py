# Generates penny-stock ($1-$10) copies of two existing daily scanners:
#   daily_ma_support_ema21_sma50_v1.py  -> daily_penny_ma_support_ema21_sma50_v1.py
#   daily_liquidity_sweep_breakout_v1.py -> daily_penny_liquidity_sweep_breakout_v1.py
# The pattern logic is unchanged. Edits (each checked):
#   - price universe 1 <= close <= 10 (min_price 1, new max_price 10)
#   - skip stocks whose ATR14 is under 1% of price (pinned / takeover names,
#     where a stop cents away would make the $40 sizing ask for thousands of shares)
#   - "Penny ($1-10)" in titles, email subject and output file names
# Usage: python gen_penny_variants.py [outdir]
import os, sys, py_compile

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
JOBS = [
    ("daily_ma_support_ema21_sma50_v1.py", "daily_penny_ma_support_ema21_sma50_v1.py",
     "Daily MA Support", "daily_ma_support_", "📊 MA Support EMA21/SMA50 —"),
    ("daily_liquidity_sweep_breakout_v1.py", "daily_penny_liquidity_sweep_breakout_v1.py",
     "Daily Liquidity Sweep Breakout", "daily_liquidity_sweep_breakout_", "📊 Liquidity Sweep Breakout —"),
]

for src, out, title, fprefix, subj in JOBS:
    s = open(src).read()

    def rep(a, b, n=1):
        global s
        k = s.count(a)
        if (n == "any" and k == 0) or (n != "any" and k != n):
            sys.exit(f"❌ {src}: expected {n} match(es), found {k}: {a[:70]!r}")
        s = s.replace(a, b)

    rep('    "min_price"              : 10.0,\n',
        '    "min_price"              : 1.0,    # penny universe: $1 - $10\n'
        '    "max_price"              : 10.0,\n'
        '    "min_atr_pct"            : 1.0,    # skip stocks moving < 1% a day (e.g. pinned by a takeover)\n')
    rep('    if price   < CFG["min_price"]:      return None\n',
        '    if price   < CFG["min_price"]:      return None\n'
        '    if price   > CFG["max_price"]:      return None\n'
        '    _h, _l, _c = (df_daily[x].values.astype(float) for x in ("High", "Low", "Close"))\n'
        '    _pc = np.concatenate([[np.nan], _c[:-1]])\n'
        '    _tr = np.nanmax(np.vstack([_h - _l, np.abs(_h - _pc), np.abs(_l - _pc)]), axis=0)\n'
        '    _atr = float(np.nanmean(_tr[-14:]))\n'
        '    if not np.isfinite(_atr) or _atr / price * 100 < CFG["min_atr_pct"]: return None\n')
    rep(title, title.replace("Daily ", "Daily Penny ($1-10) "), "any")
    rep(f'f"{fprefix}', f'f"{fprefix.replace("daily_", "daily_penny_")}', "any")
    rep(f'f"tv_{fprefix}', f'f"tv_{fprefix.replace("daily_", "daily_penny_")}')
    rep(f'subj = (f"{subj}', f'subj = (f"{subj.replace("📊 ", "📊 PENNY ")}')
    rep("price >= 10 USD", "price 1-10 USD, ATR >= 1% of price", "any")
    rep('min_price                      10 → 5', 'min_atr_pct                   1.0 → 0.5', "any")

    path = os.path.join(OUTDIR, out)
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"✅ wrote {path}")
