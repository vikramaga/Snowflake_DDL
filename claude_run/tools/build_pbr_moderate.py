# One-off build run by ClaudeRunner: the daily pullback JMA reclaim scanner
# now also accepts a moderate reclaim candle (body 0.4-0.6 ATR) and labels
# each match Strong (>= 0.6 ATR) or Moderate. Applies the edits, checks the
# result against the locally tested copy (SHA-256), commits, then runs the
# scanner once so today's email goes out with the new rule.
import hashlib, subprocess, sys

F = "daily_pullback_jma_reclaim_v1.py"
WANT = "2860bac428b66bc9fa62f6b888474e5ab85bed55d2f88b4707dbfa414f144be5"
EDITS = [
 ('#   5. RECLAIM CANDLE: green, body >= 0.6 ATR(14), closes back above the JMA\n#      (the day before closed at or below it) and above EMA21.\n',
  '#   5. RECLAIM CANDLE: green, body >= 0.4 ATR(14), closes back above the JMA\n#      (the day before closed at or below it) and above EMA21. Labelled\n#      "Strong" when the body is >= 0.6 ATR, "Moderate" when 0.4-0.6 ATR.\n'),
 ('    "min_body_atr"            : 0.6,\n',
  '    "min_body_atr"            : 0.4,   # moderate reclaim accepted\n    "strong_body_atr"         : 0.6,   # body >= 0.6 ATR = "Strong"\n'),
 ('    flags = [f"pullback {sig[\'pullback_days\']}d to {sig[\'low_vs_sma50\']:+.1f}% vs SMA50",\n             f"reclaim body {sig[\'body_atr\']:.1f} ATR", f"vol {vol_x:.1f}x"]\n',
  '    strength = "Strong" if sig["body_atr"] >= CFG["strong_body_atr"] else "Moderate"\n    flags = [f"{strength} reclaim ({sig[\'body_atr\']:.1f} ATR)",\n             f"pullback {sig[\'pullback_days\']}d to {sig[\'low_vs_sma50\']:+.1f}% vs SMA50", f"vol {vol_x:.1f}x"]\n'),
 ('        "Reclaim_Body_ATR": round(sig["body_atr"], 2), "Reclaim_Volume_x": round(vol_x, 2),\n',
  '        "Reclaim_Strength": strength,\n        "Reclaim_Body_ATR": round(sig["body_atr"], 2), "Reclaim_Volume_x": round(vol_x, 2),\n'),
 ('        "Low_vs_SMA50_%","Depth_ATR","Reclaim_Body_ATR","Reclaim_Volume_x",\n',
  '        "Low_vs_SMA50_%","Depth_ATR","Reclaim_Strength","Reclaim_Body_ATR","Reclaim_Volume_x",\n'),
 ('              "Target_1_Swing_High","T1_R","Pullback_%","Low_vs_SMA50_%","Reclaim_Body_ATR",\n',
  '              "Target_1_Swing_High","T1_R","Pullback_%","Low_vs_SMA50_%","Reclaim_Strength","Reclaim_Body_ATR",\n'),
 ('        print(f"  ✅ {sym:<6} ${r[\'Price\']:.2f}  Score {r[\'Score\']}  pullback {r[\'Pullback_%\']}% "\n              f"({r[\'Low_vs_SMA50_%\']:+}% vs SMA50)  body {r[\'Reclaim_Body_ATR\']} ATR  {r[\'Signal_Date\']}")\n',
  '        print(f"  ✅ {sym:<6} ${r[\'Price\']:.2f}  Score {r[\'Score\']}  pullback {r[\'Pullback_%\']}% "\n              f"({r[\'Low_vs_SMA50_%\']:+}% vs SMA50)  {r[\'Reclaim_Strength\']} body {r[\'Reclaim_Body_ATR\']} ATR  {r[\'Signal_Date\']}")\n'),
 ('                         f"pullback {r[\'Pullback_%\']}% ({r[\'Low_vs_SMA50_%\']:+}% vs SMA50)  {r[\'Signal_Date\']}")\n',
  '                         f"pullback {r[\'Pullback_%\']}% ({r[\'Low_vs_SMA50_%\']:+}% vs SMA50)  {r[\'Reclaim_Strength\']}  {r[\'Signal_Date\']}")\n'),
 ('                if c == "Target_1_Swing_High": sty = "color:#2563eb;font-weight:600"\n',
  '                if c == "Target_1_Swing_High": sty = "color:#2563eb;font-weight:600"\n                if c == "Reclaim_Strength": sty = "font-weight:700;color:" + ("#16a34a" if r.get(c) == "Strong" else "#b45309")\n'),
 ('then a strong green candle back above the JMA (the RVTY 14 Sep pattern)</p>',
  'then a green candle back above the JMA — Strong (body &ge; {CFG[\'strong_body_atr\']} ATR) or Moderate (&ge; {CFG[\'min_body_atr\']} ATR) (the RVTY 14 Sep pattern)</p>'),
 ("  5) Signal day: green, body >= {CFG['min_body_atr']} ATR, closes back above the JMA and EMA{CFG['ema_period']}\n",
  "  5) Signal day: green, body >= {CFG['min_body_atr']} ATR (Strong >= {CFG['strong_body_atr']} ATR, else Moderate),\n     closes back above the JMA and EMA{CFG['ema_period']}\n"),
 ("  ⚙️  If 0 results, check the FUNNEL: relax min_body_atr 0.6 → 0.4,\n     zone_tolerance_pct 0.5 → 1.5, or signal_lookback_days 2 → 5\n",
  "  ⚙️  If 0 results, check the FUNNEL: relax zone_tolerance_pct 0.5 → 1.5,\n     min_depth_atr 1.5 → 1.0, or signal_lookback_days 2 → 5\n"),
]
s = open(F).read()
if '"strong_body_atr"' not in s:
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"❌ expected 1 match: {old[:70]!r}")
        s = s.replace(old, new)
    open(F, "w").write(s)
got = hashlib.sha256(open(F, "rb").read()).hexdigest()
print(("OK  " if got == WANT else "BAD ") + F, got)
if got != WANT:
    sys.exit("❌ differs from the tested copy — nothing committed")

def git(*a):
    print("+ git", " ".join(a)); return subprocess.call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", F)
if git("diff", "--cached", "--quiet") != 0:
    git("commit", "-m", """Pullback JMA reclaim: accept moderate reclaim candles, label Strong / Moderate

Step 5 now needs a green body >= 0.4 ATR14 (was 0.6). Matches with a body
>= 0.6 ATR are labelled Strong, 0.4-0.6 ATR Moderate (new Reclaim_Strength
column in the CSV and email, and in the flags).

Verified: with the old 0.6 threshold the rule still equals the calibrated
watchlist check (0 mismatches on 12,080 synthetic days); at 0.4 the signals
are a superset (4 -> 20); a 0.5 ATR reclaim passes as Moderate, a 1.8 ATR one
as Strong, a 0.3 ATR one is rejected; mocked full runs with and without matches.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
    for i in range(5):
        if git("pull", "--rebase", "origin", "master") == 0 and git("push", "origin", "HEAD:master") == 0:
            print("✅ pushed"); break
    else:
        sys.exit("❌ push failed")
sys.exit(subprocess.call([sys.executable, F]))
