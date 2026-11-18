# One-off patch run by ClaudeRunner (2026-11-18): daily + weekly RSI bullish divergence scanners
# rebuilt to the "RSI Divergence + Long Entry" chart - signal fires on the exact entry candle:
# lower low in price + higher low in RSI (first low near oversold), then the FIRST green
# confirmation close after the lower low. Stop below the lower low, target the previous swing high.
# Rewrites both files, checks each result's SHA-256, commits, then runs both scanners.
import hashlib, py_compile, re, subprocess, sys

TF = {
    "daily_rsi_bullish_divergence_v1.py": dict(
        tf="Daily", unit="session", units="sessions", df="df_daily", base="df_daily", days_key="Days_Since_Swing_Low",
        cfg_head='''    "history_days"          : 550,   # daily bars - ~390 trading days
''',
        arm=3, window=150, maxl2=3, l1min=5, l1max=60, rsi1=35, newlow=20, high=30,
        want="7eec2e78f7196ab326feb881b4a319cb984c873d0d8084d7e2a301ea440a52e7"),
    "weekly_rsi_bullish_divergence_v1.py": dict(
        tf="Weekly", unit="week", units="weeks", df="df_weekly", base="df_daily", days_key="Weeks_Since_Swing_Low",
        cfg_head='''    "history_days"          : 900,   # daily bars, resampled to ~129 weekly bars
''',
        arm=2, window=60, maxl2=2, l1min=3, l1max=30, rsi1=40, newlow=12, high=20,
        want="f5f003ba1e803a483af428e1e0cd62885bebb6fbc21a9e9851da61e1bf2c0385"),
}

HEADER = '''# ============================================================
# NASDAQ - {tf} RSI Bullish Divergence + Long Entry (v2)
# ============================================================
#
# Rebuilt 2026-11-18 from the "RSI Divergence + Long Entry" chart: the
# signal fires on the EXACT ENTRY CANDLE ({unit} bars{wk_note}).
#
#   1. LOWER LOW: a new price low (L2) made 1-{maxl2} {units} before the
#      latest bar - the lowest low of the last {newlow} {units} - that is
#      LOWER than an earlier confirmed swing low (L1: lower than the
#      {arm} bars on each side) {l1min}-{l1max} {units} before it, with no
#      low in between below L2.
#   2. RSI HIGHER LOW: RSI(14) at L2 is HIGHER than RSI at L1, and RSI at
#      L1 was oversold / near it (<= {rsi1}).
#   3. CONFIRMATION CLOSE = ENTRY: the latest bar is the FIRST confirmation
#      candle after L2 - green (close > open), close above the previous
#      close, close in the upper half of its range - and nothing after L2
#      traded below L2. If an earlier bar after L2 already confirmed, the
#      entry has passed and the stock is not listed.
#
# Entry  = the confirmation close (latest bar)
# Stop   = below the lower low: L2 - 0.1 ATR(14)
# Target = the previous swing high: the highest high of the {high} {units}
#          up to L1 (where the decline began). Reward_R = (Target - Entry) / risk.
#
# DATA - 1 download per ticker: daily bars{wk_data}.
# SINGLE PASS - purely technical, no fundamentals fetch.
#
# ============================================================

'''

CFG_BLOCK = '''CFG = {{
{cfg_head}
    # -- Pattern (bars = {units}) ----------------------------------
    "rsi_period"              : 14,
    "swing_arm"               : {arm},   # L1 = a low lower than this many bars on each side
    "swing_search_window"     : {window},  # bars kept for the RSI / swing search
    "max_bars_since_swing_low": {maxl2},   # lower low (L2) made 1-{maxl2} bars before the entry bar
    "l1_min_gap"              : {l1min},   # L1 at least this many bars before L2 ...
    "l1_max_gap"              : {l1max},  # ... and at most this many
    "l1_rsi_max"              : {rsi1},  # RSI at L1 oversold / near it
    "new_low_lookback"        : {newlow},  # L2 = lowest low of this many bars
    "swing_high_lookback"     : {high},  # target = highest high of this many bars up to L1
    "stop_buffer_atr"         : 0.1,  # stop = L2 - 0.1 ATR(14)

    # -- Filters ---------------------------------------------------
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}}
'''

LOGIC = '''def find_swing_lows(lows, arm, start, end):
    """Bar indices p in [start, end] whose Low is lower than the `arm` bars
    before it and not above the `arm` bars after it (swing low). Oldest first."""
    out = []
    for p in range(max(arm, start), min(end, len(lows) - arm - 1) + 1):
        if lows[p] < lows[p - arm:p].min() and lows[p] <= lows[p + 1:p + arm + 1].min():
            out.append(p)
    return out

def _confirm(o, h, l, c, k):
    """Confirmation candle: green, close above the previous close, close in the upper half."""
    rng = h[k] - l[k]
    return c[k] > o[k] and c[k] > c[k - 1] and (rng <= 0 or (c[k] - l[k]) / rng >= 0.5)

def check_divergence(df, rsi, cfg):
    """
    SINGLE SOURCE OF TRUTH for the pattern. Evaluated on the latest bar i:
      L2 = lower low 1..max_bars_since_swing_low bars before i,
      L1 = earlier confirmed swing low with a higher price but lower RSI,
      bar i = FIRST confirmation close after L2 (the entry).
    Returns (passed, details); details always holds each stage's boolean.
    """
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    r = rsi.values.astype(float)
    n = len(df); i = n - 1
    det = {"price_broke_low": False, "rsi_held_higher": False, "fresh": False}
    K = cfg["max_bars_since_swing_low"]
    if n < cfg["new_low_lookback"] + cfg["l1_max_gap"] + 10: return False, det
    win = l[i - K:i]
    j = i - K + int(np.argmin(win))                                   # lower low L2
    if not (l[j] <= l[max(0, j - cfg["new_low_lookback"] + 1):j + 1].min()): return False, det
    if l[j + 1:i + 1].min() <= l[j]: return False, det                # nothing below L2 since
    pivots = find_swing_lows(l, cfg["swing_arm"], j - cfg["l1_max_gap"], j - cfg["l1_min_gap"])
    pivots = [p for p in pivots if p + cfg["swing_arm"] < j and l[j] < l[p] and l[p + 1:j].min() > l[j]]
    if not pivots: return False, det
    det["price_broke_low"] = True
    pick = None
    for p in reversed(pivots):                                        # most recent L1 first
        if np.isfinite(r[p]) and np.isfinite(r[j]) and r[j] > r[p] and r[p] <= cfg["l1_rsi_max"]:
            pick = p; break
    if pick is None: return False, det
    det["rsi_held_higher"] = True
    p = pick
    first = _confirm(o, h, l, c, i) and not any(_confirm(o, h, l, c, k) for k in range(j + 1, i))
    det["fresh"] = bool(first)
    det.update({
        "i1": p, "i2": j, "low1": float(l[p]), "low2": float(l[j]), "rsi1": float(r[p]), "rsi2": float(r[j]),
        "date1": df.index[p], "date2": df.index[j], "bars_since_swing_low": i - j,
        "swing_high": float(h[max(0, p - cfg["swing_high_lookback"]):p + 1].max()),
        "swing_high_date": df.index[max(0, p - cfg["swing_high_lookback"]) +
                                    int(np.argmax(h[max(0, p - cfg["swing_high_lookback"]):p + 1]))],
        "entry_date": df.index[i],
    })
    return bool(first), det

FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_price_broke_low": 0,
    "passed_step2_rsi_held_higher": 0,
    "passed_step3_fresh": 0,
}

def analyze_rsi_divergence(sym, df_daily):
    """Returns the entry-candle setup dict, or None."""
    if df_daily is None:
        return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
@@resample@@    rsi = calc_rsi(@@df@@["Close"], CFG["rsi_period"])

    global FUNNEL_COUNTS
    FUNNEL_COUNTS["tickers_checked"] += 1
    passed, sig = check_divergence(@@df@@, rsi, CFG)
    if sig.get("price_broke_low"):
        FUNNEL_COUNTS["passed_step1_price_broke_low"] += 1
        if sig.get("rsi_held_higher"):
            FUNNEL_COUNTS["passed_step2_rsi_held_higher"] += 1
            if sig.get("fresh"):
                FUNNEL_COUNTS["passed_step3_fresh"] += 1
    if not passed:
        return None

    hh, ll, cc = (@@df@@[k].values.astype(float) for k in ("High", "Low", "Close"))
    pc = np.concatenate([[np.nan], cc[:-1]])
    tr = np.nanmax(np.vstack([hh - ll, np.abs(hh - pc), np.abs(ll - pc)]), axis=0)
    atr = float(pd.Series(tr).rolling(14).mean().values[-1])
    entry_price = float(cc[-1])
    stop_loss   = sig["low2"] - CFG["stop_buffer_atr"] * (atr if np.isfinite(atr) else 0)
    if entry_price <= stop_loss:
        return None
    risk = entry_price - stop_loss
    risk_pct = risk / entry_price * 100
    target = sig["swing_high"]
    reward_r = (target - entry_price) / risk
    price_break_pct = (sig["low1"] - sig["low2"]) / sig["low1"] * 100 if sig["low1"] > 0 else 0
    rsi_divergence_pts = sig["rsi2"] - sig["rsi1"]
    bounce_pct = (entry_price - sig["low2"]) / sig["low2"] * 100 if sig["low2"] > 0 else 0

    score = 25
    reasons = []
    score += min(25, rsi_divergence_pts * 1.5); reasons.append(f"RSIDiverge+{rsi_divergence_pts:.1f}")
    score += min(15, price_break_pct * 5);      reasons.append(f"PriceBreak-{price_break_pct:.1f}%")
    score += min(20, max(0, reward_r) * 5);      reasons.append(f"Target {reward_r:.1f}R")
    score += max(0, min(15, (CFG["l1_rsi_max"] - sig["rsi1"]) * 1.5)); reasons.append(f"L1 RSI {sig['rsi1']:.1f}")
    reasons.append(f"lower low {sig['bars_since_swing_low']} @@unit@@(s) before entry")
@@wk_flag@@    if reward_r < 1: reasons.append("target < 1R")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Target"         : round(target, 2),
        "Reward_R"       : round(reward_r, 2),
        "Entry_Date"     : sig["entry_date"].strftime("%Y-%m-%d"),
        "Swing_High_Date": sig["swing_high_date"].strftime("%Y-%m-%d"),
        "RSI_At_Low2"    : round(sig["rsi2"], 1),
        "RSI_At_Low1"    : round(sig["rsi1"], 1),
        "RSI_Divergence_Pts": round(rsi_divergence_pts, 1),
        "Price_Low2"     : round(sig["low2"], 2),
        "Price_Low1"     : round(sig["low1"], 2),
        "Price_Break_%"  : round(price_break_pct, 1),
        "Bounce_From_Low_%": round(bounce_pct, 1),
        "Swing_Low_Date" : sig["date2"].strftime("%Y-%m-%d"),
        "Prior_Swing_Low_Date": sig["date1"].strftime("%Y-%m-%d"),
        "@@days_key@@": sig["bars_since_swing_low"],
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
@@wk_df@@        "_rsi"           : rsi,
    }

'''

FOOTER = '''  📋 SIGNAL - "RSI Divergence + Long Entry" ({tf} bars). Fires on the EXACT
  entry candle: the latest {unit} bar is the first confirmation close.

  1) LOWER LOW: L2 = a new low 1-{{CFG['max_bars_since_swing_low']}} {units} before the latest bar
     (lowest low of {{CFG['new_low_lookback']}} {units}), LOWER than an earlier swing low L1
     ({{CFG['l1_min_gap']}}-{{CFG['l1_max_gap']}} {units} before it; lower than {{CFG['swing_arm']}} bars each side).
  2) RSI HIGHER LOW: RSI(14) at L2 > RSI at L1, and RSI at L1 <= {{CFG['l1_rsi_max']}}.
  3) CONFIRMATION CLOSE = ENTRY: the latest bar is green, closes above the
     previous close and in the upper half of its range, it is the FIRST such
     candle since L2, and nothing traded below L2 since.

  📋 OUTPUT
  Entry_Price = the confirmation close (Entry_Date)
  Stop_Loss   = L2 - {{CFG['stop_buffer_atr']}} ATR(14)  (below the lower low)
  Target      = previous swing high (highest high of {{CFG['swing_high_lookback']}} {units} up to L1)
  Reward_R    = (Target - Entry) / (Entry - Stop)
  Swing_Low_Date / Prior_Swing_Low_Date = L2 / L1

  📋 SCORE (0-100)
  25 base + RSI divergence (0-25) + depth of the lower low (0-15)
  + Reward_R (0-20) + how oversold L1 was (0-15)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
'''

def fill(t, d):
    for k, v in d.items(): t = t.replace("@@" + k + "@@", str(v))
    return t

def sub_once(s, old, new, label, count=1):
    n = s.count(old)
    if (count == 1 and n != 1) or (count == "+" and n < 1):
        sys.exit(f"{label}: expected {count} match, found {n} - nothing changed")
    return s.replace(old, new)

def region(s, start, end, new, label, keep_end=True):
    a = s.find(start)
    if a < 0: sys.exit(f"{label}: start marker missing")
    b = s.find(end, a + len(start))
    if b < 0: sys.exit(f"{label}: end marker missing")
    return s[:a] + new + (s[b:] if keep_end else s[b + len(end):])

out = {}
for F, P in TF.items():
    s = open(F, encoding="utf-8").read()
    wk = P["tf"] == "Weekly"
    fmt = dict(P, wk_note="; weekly bars are built from daily bars and the current week counts while in progress" if wk else "",
               wk_data=", resampled to weekly ('W')" if wk else "")
    s = region(s, "# ====", "import subprocess, sys, os", HEADER.format(**fmt), "header")
    s = region(s, "CFG = {", "\n}\n", CFG_BLOCK.format(**fmt), "cfg", keep_end=False)
    resample = ('    df_weekly = resample_ohlcv(df_daily, "W")\n'
                '    if len(df_weekly) < CFG["rsi_period"] + CFG["l1_max_gap"] + CFG["new_low_lookback"] + 10:\n'
                '        return None\n') if wk else \
               ('    if len(df_daily) < CFG["rsi_period"] + CFG["l1_max_gap"] + CFG["new_low_lookback"] + 20:\n'
                '        return None\n')
    wk_flag = ('    if df_daily.index[-1].weekday() < 4: reasons.append("week in progress - confirms at the weekly close")\n'
               if wk else "")
    wk_df = '        "_df_weekly"     : df_weekly,\n' if wk else ""
    s = region(s, "def find_swing_lows", "# ── Download: daily",
               fill(LOGIC, dict(fmt, resample=resample, wk_flag=wk_flag, wk_df=wk_df)), "logic")
    # funnel labels
    s = re.sub(r"Step 1 — price broke its last swing low\s*:", "Step 1 — price made a lower low (L2 < L1)   :", s)
    s = re.sub(r"Step 2 — \+ RSI held above its last swing low :", "Step 2 — + RSI higher low, L1 RSI oversold   :", s)
    s = re.sub(r"Step 3 — \+ fresh \(within \{CFG\['max_bars_since_swing_low'\]\}\w\)\s*:",
               "Step 3 — + confirmation close today (entry)  :", s)
    s = sub_once(s, "(evaluated once per liquid ticker, on its 2 most recent confirmed swing lows)",
                 "(evaluated once per liquid ticker on its latest bar)", "funnel note")
    s = s.replace("passed Step 1 (price broke low)", "passed Step 1 (lower low)")
    s = s.replace("passed Step 2 (RSI held higher)", "passed Step 2 (RSI higher low)")
    s = s.replace("passed Step 3 (fresh) = full match", "passed Step 3 (confirmation close today) = entry signal")
    s = s.replace("passed price-broke-low -> ", "passed lower-low -> ")
    s = s.replace("passed rsi-held-higher -> ", "passed rsi-higher-low -> ")
    s = s.replace("passed fresh (=full match)", "passed confirmation close today (=entry)")
    s = s.replace("matches (price broke its last swing low, RSI held above its last swing low)",
                  "entry signals (lower low + RSI higher low, confirmation close today)")
    s = re.sub(r"found — red, red \(higher low\), green\n\s*price broke its last swing low, RSI did not",
               "found — lower low + RSI higher low, confirmation close today = entry", s)
    # columns / email
    s = sub_once(s, '"Entry_Price","Stop_Loss","Risk_%",\n',
                 '"Entry_Price","Stop_Loss","Risk_%","Target","Reward_R","Entry_Date","Swing_High_Date",\n', "COLS")
    s = sub_once(s, '''            for c in ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                      "RSI_At_Low2","RSI_At_Low1","RSI_Divergence_Pts","Swing_Low_Date"]''',
                 '''            for c in ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target","Reward_R",
                      "RSI_At_Low2","RSI_At_Low1","RSI_Divergence_Pts","Swing_Low_Date"]''', "email head")
    s = sub_once(s, '''f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
''', '''f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6">${float(r.get("Target",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(r.get("Reward_R",0) or 0):.1f}R</td>'
''', "email row")
    s = sub_once(s, "<td colspan=\"9\"", "<td colspan=\"11\"", "colspan")
    s = sub_once(s, '''f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "''',
                 '''f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  Target:${float(r.get('Target',0) or 0):.2f} ({float(r.get('Reward_R',0) or 0):.1f}R)  "''',
                 "plain row")
    s = sub_once(s, '''        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls=":", alpha=0.7,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
''', '''        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls=":", alpha=0.7,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.axhline(r["Target"], color="#3b82f6", lw=1.0, ls="--", alpha=0.8,
                  label=f"Target ${r['Target']:.2f}")
        ax.axhline(r["Entry_Price"], color="#22c55e", lw=0.8, ls="-", alpha=0.6,
                  label=f"Entry ${r['Entry_Price']:.2f}")
''', "chart lines")
    # footer + tuning text
    s = region(s, "  📋 SIGNAL (all required", '""")', FOOTER.format(**fmt)[:-len('""")\n')], "footer")
    s = re.sub(r'    print\("   max_bars_since_swing_low .*?\n    print\("   swing_arm .*?\n    print\("   swing_search_window .*?\n',
               '    print("   l1_rsi_max        raise (e.g. 35 -> 40): accept a less oversold first low")\n'
               '    print("   max_bars_since_swing_low  2-3 -> 5: allow more bars between the lower low and the entry")\n',
               s, flags=re.S)
    s = s.replace("swing low confirmed within the last {CFG['max_bars_since_swing_low']} trading days",
                  "entry = first confirmation close after the lower low")
    s = s.replace("swing low confirmed within the last {CFG['max_bars_since_swing_low']} weeks",
                  "entry = first confirmation close after the lower low")
    s = s.replace("  A stock only matches if the divergence and freshness conditions",
                  "  A stock only matches on the bar that confirms the divergence entry")
    s = s.replace('print("  and SMA structure all fire within the lookback window\\n")',
                  'print("  (the first confirmation close after the lower low)\\n")')
    s = s.replace("depth + freshness + initial bounce strength)", "depth + Reward_R + how oversold L1 was)")
    s = s.replace("Stop_Loss       the most recent swing low's price", "Stop_Loss       below the lower low (L2 - 0.1 ATR)")
    s = s.replace("Entry_Price     today's close", "Entry_Price     the confirmation close (latest bar)")
    out[F] = s

if "--dry" in sys.argv:
    for F, s in out.items(): open(F + ".new", "w", encoding="utf-8").write(s)
    sys.exit(0)
for F, s in out.items():
    open(F, "w", encoding="utf-8").write(s)
    got = hashlib.sha256(s.encode("utf-8")).hexdigest()
    if got != TF[F]["want"]: sys.exit(f"{F}: sha mismatch {got} != {TF[F]['want']}")
    py_compile.compile(F, doraise=True)
    print("patched", F, got)

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *TF)
git("commit", "-m", """RSI divergence (daily + weekly): signal on the exact entry candle

Lower low + RSI higher low (first low near oversold), first confirmation
close after the lower low = entry; stop below the lower low, target the
previous swing high.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for _ in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("pushed"); break
else:
    sys.exit("push failed")
rc = 0
for F in TF:
    print(f"\n===== running {F} =====", flush=True)
    rc |= subprocess.call([sys.executable, F])
sys.exit(rc)
