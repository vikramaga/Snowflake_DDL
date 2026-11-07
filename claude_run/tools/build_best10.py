# One-off build run by ClaudeRunner: applies claude_run/tools/patch_best10.py
# (Best 10 shortlist in every scheduled scanner's email, via best_picks.py),
# checks every patched file is byte-identical to the locally tested copy
# (SHA-256) and commits. Nothing is committed if any file differs.
import hashlib, subprocess, sys

EXPECTED = {
    "daily_price_jma_ema21_sma50_volume_v1.py": "02a6791c553776e43e28a06f35b0eff4bf2b384cebc71933582d2f810fc765e8",
    "daily_3candle_jma_reclaim_v1.py": "68e28631a015341447143d74e38cd37dd955c7f90f2a586bd421e409ae964dc5",
    "daily_liquidity_sweep_breakout_v1.py": "4ee8cc81f0d487df8c28c56dc867eec4077bbe8902c99eda531160cb0cc76f49",
    "daily_rsi_bullish_divergence_v1.py": "1952415a4a732a3d31e3d589f857c3e720b9c38d396312d3c39eebc572451728",
    "daily_rsi_high_breakout_v1.py": "2e6c9423bbb64ff679709782c331529006753289535698e1b3e573cb845154af",
    "daily_3candle_higherlow_above_sma50_v1.py": "8be98d8e536f529cb82175093e53a68d2def493efefebca5068030adbf295260",
    "weekly_rsi_bullish_divergence_v1.py": "1b1c052dc03a7352a7f1a667b0b16d3890f92f171c8cacc119235f9b2a1d39b9",
    "weekly_reversal_2x_above_sma50_v1.py": "d4003897faf99b0d0a1cbd507d66786cb58a7a9a8c42325def75c26e3841dc2a",
    "monthly_reversal_2x_above_sma50_v1.py": "47c6b45898e57793a996cc9a39c78ac44d9de30b5850a527c5a8b0f2ce8bd788",
    "monthly_rsi_cross_above_avg_v1.py": "e24e6f3e3277858a5e0133f6d6a0e3ad152625cbb7a391e85b59812330c28040",
    "monthly_3candle_reclaim_jma_ema8_v1.py": "2d14fc8359835574e57aca4a653e16dce7a0ff27eec6c4d452bed964002685a6",
    "india_weekly_reversal_2x_above_sma50_v1.py": "75afa39d82b300689016ae55fe5833bfdf6e42d22593b40463a63ee4c3b57426",
    "daily_ma_support_ema21_sma50_v1.py": "3339def8a2bb64ca201b559fbc05eb9885ec1dec5a577366cda4e93cb1a5fa3b",
    "daily_jma_cross_ema8_v1.py": "647375c2054bf56f4a3d5ea6d302ae141a6cf624de62e196ad0f2c939ff0aaf9",
    "daily_sweep_sma50_strategy_v1.py": "af265b29c9e744834fa3fd4da718108f3fcb22ff8e4803545698da4a6488d308",
    "daily_rsi_avg_bounce_v1.py": "3c0e5549885e6077deccb50d09c111b046bfbe0902a9255353c3825d3c61fea5",
    "daily_rsi2_snapback_v1.py": "834390b7e9673f655615a0ec530bf73fa3a79c1b4ab6795926de3f5ec10cf391",
    "daily_ma_sweep_4x_v1.py": "f707899bd16304f347452aac925721dcf2475835f0c65ffc544f734e306ec4d6",
    "daily_jma_reclaim_rsi_v1.py": "2ea27f58c1708b60f1beeb87908f9248ae7201a0565c2fc480884c79bb4d3e5e",
    "daily_penny_ma_support_ema21_sma50_v1.py": "cb17cf11c9787fa7d33c66319583aebdd581d823851401d7fdb4b9c8dc043cc6",
    "daily_penny_liquidity_sweep_breakout_v1.py": "13f47d5cb26afdfcf50d450261d35d7da0a632bc246b560c82d99c5a766f5dc0",
    "daily_penny_jma_reclaim_rsi_v1.py": "aac8cdc6151d3b96404ef0f9d376be617cef88424880837d1d0a53da706f7260",
    "monthly_rsi_weekly_green_reversal_v1.py": "2acf04ddb621cdd5e945511bd08dc2edec7f7324d3fa4fd2bbcf71b164e25dd4",
    "weekly_doji_jma_reclaim_v1.py": "708b127dc6636b4417a384864aedace0c14fc0e7fa4cc13eb2eb9878d7551770",
    "rsi_avg_turn_up_v1.py": "afdf65e1774228ce75a5ebc0ba1bffaf738510a8114924a2feb2c0d0a5b45b93",
    "daily_doji_hammer_jma_reclaim_v1.py": "cdcfae217215753aae2e8311c5d00d15d3779323865e4b3ff66c076933511460",
    "daily_sma50_150_squeeze_turn_v1.py": "79bd623b29d5618b7b6ed59d4afdd443cb2cbac4ec80bb80fdf851f890c28227",
    "daily_rsi2_rs_1to1_v1.py": "e3a1d59e91f28c7021544246a80c65a82c35bfcdc75ec5fe4082c6b8acf51189",
    "claude_run/tools/rsi_amount_filter.py": "ce0a36227dfaccc8b44cd0123d21e9ae737b02cd4e10afacf3ad0b5d4255aa96",
    "monday_red_candle_breakout_v1.py": "e433b0b9bcd7925ef6f96f1a728b9b9b7d67d41e0751bee0ebfa032a1ca39d25",
}
subprocess.check_call([sys.executable, "claude_run/tools/patch_best10.py", "."])
bad = []
for f, want in EXPECTED.items():
    got = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print(("OK  " if got == want else "BAD ") + f)
    if got != want:
        bad.append(f)
if bad:
    sys.exit(f"❌ {len(bad)} file(s) differ from the tested copies — nothing committed")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *EXPECTED.keys())
git("commit", "-m", """Add a Best 10 shortlist to every scheduled scanner's email

Each of the 28 scheduled scanners now calls best_picks.rank_best(df_out) after
position sizing and best_picks.attach_to_email(msg) before sending. The email
opens with a Best 10 table (full list unchanged below), the subject names the
top 3, and best10_<scanner>_<ts>.csv is attached and saved with the results.

Best score 0-100 = Trend 35 (above SMA200, SMA50 rising, SMA200 rising /
turning up) + Catalyst 30 (contract/order and positive headlines in 30 days,
net analyst upgrades in 60 days, last EPS beat; negative headlines subtract)
+ Value 35 (analyst target upside, forward P/E or PEG, Buy rating, entry not
extended above SMA50/SMA200). Data from Yahoo via yfinance; every step is
guarded so a data problem cannot stop a scanner or its email.

The RSI digest and the Monday breakout scanner now ignore best10_*.csv.

Applied by claude_run/tools/patch_best10.py (checked edits, idempotent).
Verified: all 28 patched scanners run end to end on synthetic data with a
mocked SMTP server, each sends one email with the Best 10 block in the HTML and
text parts (CSV attached when there are matches); scoring re-checked by hand
(contract/positive news, upgrades, EPS beat, stale news ignored, negative news,
trend points); empty results and a Yahoo outage handled.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
