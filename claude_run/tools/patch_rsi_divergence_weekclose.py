# One-off patch run by ClaudeRunner (2026-11-18): RSI divergence follow-up.
# Weekly: only COMPLETED weeks are checked (the in-progress week is dropped), so the weekly
# entry is the weekly close. Both: skip setups whose target is not above the entry or whose
# stop is more than 30% away (bad / split-distorted data).
import hashlib, py_compile, subprocess, sys
WANT = {"daily_rsi_bullish_divergence_v1.py": "7d166bdfc136bf9770169ed5034021ef7b48efcada020699479fb902fd3958c9", "weekly_rsi_bullish_divergence_v1.py": "7589af39edc1f63b0f4db4f28f247ae723fea96abda15330cafc6ab204d94df7"}

def sub_once(s, old, new, label):
    n = s.count(old)
    if n != 1: sys.exit(f"{label}: expected 1 match, found {n} - nothing changed")
    return s.replace(old, new)

SANITY_OLD = '''    risk = entry_price - stop_loss
    risk_pct = risk / entry_price * 100
    target = sig["swing_high"]
    reward_r = (target - entry_price) / risk
'''
SANITY_NEW = '''    risk = entry_price - stop_loss
    risk_pct = risk / entry_price * 100
    target = sig["swing_high"]
    reward_r = (target - entry_price) / risk
    if target <= entry_price or risk_pct > CFG["max_risk_pct"]:
        return None                       # target already passed, or a stop too far away (bad data)
'''
CFG_OLD = '''    "stop_buffer_atr"         : 0.1,  # stop = L2 - 0.1 ATR(14)
'''
CFG_NEW = '''    "stop_buffer_atr"         : 0.1,  # stop = L2 - 0.1 ATR(14)
    "max_risk_pct"            : 30,   # skip if the stop is more than 30% below the entry
'''
WK_OLD = '''    df_weekly = resample_ohlcv(df_daily, "W")
'''
WK_NEW = '''    df_weekly = resample_ohlcv(df_daily, "W")
    if df_daily.index[-1].weekday() < 4:  # week still in progress -> use completed weeks only
        df_weekly = df_weekly.iloc[:-1]
'''
FLAG_OLD = '''    if df_daily.index[-1].weekday() < 4: reasons.append("week in progress - confirms at the weekly close")
'''
FLAG_NEW = '''    reasons.append(f"weekly close {df_weekly.index[-1] - pd.Timedelta(days=2):%Y-%m-%d}")
'''
HDR_OLD = "; weekly bars are built from daily bars and the current week counts while in progress"
HDR_NEW = "; weekly bars are built from daily bars and only COMPLETED weeks are used - the entry is the weekly close"

out = {}
for F in WANT:
    s = open(F, encoding="utf-8").read()
    s = sub_once(s, SANITY_OLD, SANITY_NEW, F + " sanity")
    s = sub_once(s, CFG_OLD, CFG_NEW, F + " cfg")
    if F.startswith("weekly"):
        s = sub_once(s, WK_OLD, WK_NEW, F + " weekly close")
        s = sub_once(s, FLAG_OLD, FLAG_NEW, F + " flag")
        s = sub_once(s, HDR_OLD, HDR_NEW, F + " header")
    out[F] = s
if "--dry" in sys.argv:
    for F, s in out.items(): open(F + ".new", "w", encoding="utf-8").write(s)
    sys.exit(0)
for F, s in out.items():
    got = hashlib.sha256(s.encode("utf-8")).hexdigest()
    if got != WANT[F]: sys.exit(f"{F}: sha mismatch {got} != {WANT[F]} - nothing written")
for F, s in out.items():
    open(F, "w", encoding="utf-8").write(s); py_compile.compile(F, doraise=True); print("patched", F)

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *WANT)
git("commit", "-m", """RSI divergence: weekly entry on completed weeks only; skip target-passed / >30% stop setups

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for _ in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("pushed"); break
else:
    sys.exit("push failed")
rc = 0
for F in WANT:
    print(f"\n===== running {F} =====", flush=True)
    rc |= subprocess.call([sys.executable, F])
sys.exit(rc)
