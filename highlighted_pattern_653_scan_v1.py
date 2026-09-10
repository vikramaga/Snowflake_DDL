# ============================================================
# NASDAQ — Highlighted Pattern Scanner, scoped to a fixed 653-ticker
# list (v1) — from the Master Technical Scanner's 2026-09-10 run
# ============================================================
#
# Matches the chart pattern highlighted by the user, evaluated
# generally (no JMA-vs-EMA8-specific requirement):
#
#   1. STRUCTURE: SMA50 > SMA150 (bullish structure), evaluated on
#      the test day (candle A, below).
#
#   2. CLUSTER NEAR SMA50: the average of the fast-MA cluster
#      (EMA8, JMA, SMA21) sits within near_sma50_pct% of SMA50 on
#      the test day — the fast group is pulling back to actually
#      touch/test the medium-term average, not just converging
#      with each other.
#
#   3. TEST + BREAKOUT: candle A (the test day) is RED and closed
#      BELOW the cluster average; candle B (the next day) is GREEN
#      and closed ABOVE the cluster average — a clean one-day
#      reclaim of the tested zone.
#
#   4. VOLUME: candle B's volume is higher than candle A's.
#
# Scanned over the last signal_lookback_days trading days, ranked
# by a match score (tightness of the test, volume strength,
# freshness) — the highest-scoring stocks are the best visual match
# to the highlighted chart.
#
# UNIVERSE — restricted to EXACTLY the 653 tickers from the Master
# Technical Scanner's 2026-09-10 05:37 UTC run (hardcoded list, no
# full-universe ticker fetch — faster and directly answers "out of
# those 653 tickers").
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# candle B's high; Stop = candle A's low), not explicitly requested.
#
# SINGLE PASS — purely technical, no fundamentals fetch.
#
# ============================================================

import subprocess, sys, os

def pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

pip_install("yfinance", "pandas", "numpy", "requests", "tqdm", "matplotlib")
print("✅  Dependencies installed")

def _detect_notebook():
    try:
        if "google.colab" in sys.modules: return True
        if os.environ.get("COLAB_BACKEND_VERSION"): return True
        if os.environ.get("JPY_PARENT_PID"):
            import importlib
            if importlib.util.find_spec("ipywidgets") is not None: return True
    except Exception: pass
    return False

_IN_NOTEBOOK = _detect_notebook()
from tqdm import tqdm

def display_html(h):
    if _IN_NOTEBOOK and "IPython" in sys.modules:
        try:
            sys.modules["IPython"].display.display(
                sys.modules["IPython"].display.HTML(h))
            return True
        except Exception: pass
    return False

def display(obj):
    if _IN_NOTEBOOK and "IPython" in sys.modules:
        try: sys.modules["IPython"].display.display(obj); return
        except Exception: pass
    try: print(obj.to_string())
    except Exception: print(obj)

import yfinance as yf
import pandas as pd
import numpy as np
import requests, time, warnings, io
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
env = "Colab/Jupyter" if _IN_NOTEBOOK else "Script/CI"
print(f"✅  yfinance {yf.__version__}  |  numpy {np.__version__}  |  [{env}]")

# ── Email secret diagnostic ────────────────────────────────────
_GMAIL_USER = os.environ.get("GMAIL_USER", "")
_GMAIL_PASS = os.environ.get("GMAIL_PASS", "")
_EMAIL_TO   = os.environ.get("EMAIL_TO",   "")

print()
print("━"*65)
print("  EMAIL CONFIGURATION")
print("━"*65)
if _GMAIL_USER and _GMAIL_PASS and _EMAIL_TO:
    print(f"  ✅ GMAIL_USER  : {_GMAIL_USER[:4]}***{_GMAIL_USER[-4:]}")
    print(f"  ✅ GMAIL_PASS  : {'*'*16}  ({len(_GMAIL_PASS.replace(' ',''))} chars)")
    print(f"  ✅ EMAIL_TO    : {_EMAIL_TO}")
    print(f"  ✅ Email will be sent after scan")
else:
    missing = [k for k,v in [("GMAIL_USER",_GMAIL_USER),
                               ("GMAIL_PASS",_GMAIL_PASS),
                               ("EMAIL_TO",_EMAIL_TO)] if not v]
    print(f"  ⚠️  Missing secrets: {', '.join(missing)}")
    print(f"  ℹ️  Go to: GitHub repo → Settings → Secrets → Actions")
    print(f"       Add: GMAIL_USER, GMAIL_PASS (App Password), EMAIL_TO")
    print(f"  ℹ️  Email will be SKIPPED this run")
print("━"*65)
print()

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"           : 400,   # daily bars — plenty for SMA150 + buffer

    # ── Indicator periods ────────────────────────────────────────
    "ema8_period"             : 8,
    "sma21_period"            : 21,
    "sma50_period"            : 50,
    "sma150_period"           : 150,
    "jma_period"              : 13,
    "jma_phase"               : 40,

    # ── Cluster near SMA50 (fast-MA average testing the medium-term MA) ──
    "near_sma50_pct"          : 3.0,  # max gap (as % of SMA50) between the
                                       # {EMA8,JMA,SMA21} average and SMA50

    "signal_lookback_days"    : 15,   # ~3 trading weeks — how far back to look
                                       # for the pattern, not just today

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Fixed 653-ticker universe (from the Master Technical Scanner's
#    2026-09-10 05:37 UTC run) ─────────────────────────────────────
FIXED_TICKERS = [
    "JILL","TLX","RBCAA","OPRT","NX","SOPH","BLFS","TNK","FCF","MET","HAFN","SABR",
    "SEIC","WT","CGEM","CMPS","HUM","KNSA","OPY","PRCH","RPRX","SGHC","TUSK","CFFN",
    "KGEI","MAN","NEOG","VG","VLO","BNS","CON","FLXS","KMX","PMTS","XRN","BBY","CRBG",
    "MD","MGNI","MRX","ORMP","OSPN","PBF","PLSE","WES","DK","ELV","GH","INGM","LITE",
    "OHI","AAMI","CNXN","EQH","GEO","JOYY","JRSH","MSBI","SM","TX","XMAX","ACRS","ANRO",
    "AYA","BGC","CLBK","GLBS","GNK","JXN","KINS","LILAK","MTCH","PSX","RCMT","RDVT",
    "SXT","TALO","ACCO","ACT","BBVA","BDSX","CBZ","CIB","CMBT","CRSR","DELL","FFIV",
    "FIRY","HNGE","IHD","MSGS","NWBI","RELY","SAN","SOBO","TVTX","VRSN","AGM","AOD",
    "CARE","CM","CPAY","CUE","DINO","EHC","FBLA","HPE","IVZ","MDGL","MT","TBBB","TGT",
    "ADX","AGIO","AMC","CXT","DOCN","FISI","FLNG","GAIN","MEOH","MGTX","MS","NNBR",
    "NWL","OMDA","THG","UVE","WAT","AAPL","ABNB","ALRS","ALSN","AMD","AMG","BSM","CLMT",
    "DHX","ENGS","FA","HELP","LBRX","MATX","MG","NRIX","PUBM","SGMT","TH","TKNO","TRMD",
    "YPF","ACHV","APPS","BCAL","COKE","FLYW","IMAX","INNV","LILA","MPB","OKE","SBLK",
    "SRCE","TOP","WBI","AEF","ATAI","AVAH","BAC","BHRB","CLOV","CLYM","CNK","DNTH","DT",
    "ECO","EZPW","FORR","GCT","GFF","GLNG","ISTR","LFST","LGND","MSGM","NFJ","NUE",
    "PBYI","PK","RAL","SAFT","TRV","WCC","AGEN","APLE","BCX","CAC","EOD","ETSY","FROG",
    "HOPE","IBKR","IBTA","IDT","JAZZ","LCUT","NAT","NET","NVDA","SBRA","SJM","SLDE",
    "SNX","SUN","SUNC","TWIN","VOR","WKC","ABCL","AMN","BUUU","BY","CCNE","EC","ECPG",
    "ERO","LAD","MRBK","NHC","PFG","PHVS","PRAA","PSTL","QQQX","ROKU","SB","SLP","SNEX",
    "STGW","TEVA","UNH","AM","ASND","ASX","AVT","BFH","BNY","BSAC","BWIN","CLDT","HBT",
    "INCY","KSPI","KYN","LOVE","MMI","MYE","NVT","PAG","PAYC","QMCO","REPX","SENEA",
    "STX","TRST","UFCS","BIO","BSTZ","BTSG","CBAN","COHU","CORT","CRVL","CTKB","CURR",
    "CVE","CVI","CWBC","DXPE","EDRY","EG","FCFS","FRHC","GIC","HRTG","HSIC","IFF","INVX",
    "KALU","KFY","LPG","NRC","OGN","ORRF","PBT","PNTG","RLJ","SFST","SGHT","SIMO","SNDA",
    "TBN","TECK","TWST","VNCE","VSTS","WTI","WYHG","XYZ","ACA","AEG","ANL","AVAL","AVBC",
    "BALL","BBCP","BRBS","CDNA","CHE","CVLG","CWT","DAVE","DSGN","ET","GRDN","GTE","HBM",
    "HELE","INTA","JBHT","KB","KN","MSEX","MU","NEXA","OPHC","PAYS","PDM","PGC","PGEN",
    "PKOH","RVTY","SANM","SMBK","SXC","TEN","UMC","UVSP","WDC","WNC","XNCR","ADM","AES",
    "ANGO","BIIB","BTX","CART","CHEF","CSWC","CTO","ESTC","FET","FNV","HAFC","IGD","IMO",
    "LQDT","MNKD","MTG","NBTB","OSBC","PACS","SAIC","SPNT","STT","TILE","VRTX","WRB",
    "XPRO","ADP","ANET","APA","APH","ARM","AXGN","BHB","CODI","CRCT","DAC","DDI","EQ",
    "EXPD","FRO","FSUN","HRZN","HSBC","ILMN","ING","KRT","MAKO","MDLZ","MPC","MRVL",
    "MSFT","MSM","MTRN","NDSN","NMAX","NTRA","OBIO","QRVO","QURE","ROIV","RSI","SHIP",
    "SLF","SMTI","THFF","TSM","TWLO","URGN","UTZ","VRNS","WASH","WHD","WSFS","ZD","ZIM",
    "ABM","AIZ","ALNT","AMAL","ASH","ATLC","ATNI","AUR","BEN","BRKR","BXC","BZH","C",
    "CADL","CECO","CVX","E","EVTC","EXEL","FCBC","FMNB","FTK","GBTG","GTX","HALO","HZO",
    "KRO","LGIH","LRCX","MASS","MC","NBIS","NESR","NWSA","OBE","PBI","PGNY","RBB",
    "RUSHB","SCCO","SLDB","SSL","SUNE","VCTR","VIRT","VREX","XMTR","ZM","AGNC","ARDT",
    "ARTV","ARWR","BE","BST","CBL","CHCO","CHYM","CNO","CRWD","CSCO","DSGR","DSP","EE",
    "ESQ","ETV","EWTX","FNRN","FTHY","GCMG","HTB","IOSP","IPWR","LH","MCRI","MFC","MNTK",
    "MTA","MXL","NSIT","NTRS","PAYO","PLPC","PYPD","QUAD","RFAI","SHBI","SMFG","SMTC",
    "SND","SNOW","SPG","SRRK","STLN","UBS","VBNK","ACCL","AEHR","AIFA","AWK","BANR",
    "BLX","BMO","CNI","CRDL","DDOG","ELVN","EXG","FBNC","FTH","GGB","GM","HPK","HSTM",
    "HWC","HYLN","IEP","INTC","IRDM","MITK","MTLS","NML","NUTX","OCC","PDFS","PXS","SEI",
    "TGB","TXG","ABX","AMCX","AMH","AMTB","AVIR","BAND","BBNX","BILL","CBLL","CHW","CNC",
    "DDD","FANG","GRFS","HIMX","HTO","JHX","NEO","OBK","OMER","OVID","SBCF","SLS","TRT",
    "VET","VOC","WEST","BCRX","DLTH","EFSC","ENIC","FENC","GCO","KOS","NBXG","ROG","SFL",
    "SILC","STIM","TRLV","UTF","VEON","VOYA","WSBC","XHLD","ACOG","ANIK","ANTX","BOLD",
    "DCH","EPC","FDMT","GKOS","KNX","KPLT","KYMR","NGL","ORKA","PESI","PI","RLAY","SENS",
    "SLN","TAK","WFG","ABSI","AOUT","AXSM","CRL","CYRX","DRTS","EDD","FSK","GHRS","HFRO",
    "INSM","NGNE","RVMD","SFNC","TNGX","TXO","VOD","AMWL","CBIO","CURV","FTRE","GDOT",
    "GSM","INBX","KRRO","RDI","PYXS","RAPP","TJGC","NEXT","PEB","TSAT",
]

def calc_jma(series, period=13, phase=40, power=2):
    """
    JMA (Jurik Moving Average) approximation — adaptive EMA with
    phase-based smoothing, using the corrected e2 update (steady-
    state gain 1.0, tracks price correctly).
    """
    n      = len(series)
    vals   = series.values.astype(float)
    result = np.full(n, np.nan)
    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0)
    beta  = alpha * phase_ratio
    first_valid = 0
    for i in range(n):
        if not np.isnan(vals[i]):
            first_valid = i
            break
    e0 = e1 = e2 = vals[first_valid]
    result[first_valid] = e0
    for i in range(first_valid + 1, n):
        v   = vals[i]
        e0  = (1 - alpha) * e0 + alpha * v
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (1 - alpha) * e2 + alpha * (e0 + e1)
        result[i] = e2
    return pd.Series(result, index=series.index)

def check_pattern_at(df, ema8, jma, sma21, sma50, sma150, i, cfg):
    """
    Checks the full pattern with candle B (the breakout/reclaim day)
    anchored at bar `i`. Candle A (the test day) is bar i-1.
    This is the SINGLE SOURCE OF TRUTH for the pattern logic.

    Returns (passed: bool, details: dict). details is {} only when
    there isn't enough data; otherwise it always contains every
    stage's boolean, even on failure (for diagnostics).
    """
    if i < 1 or i >= len(df):
        return False, {}

    s50, s150 = sma50.iloc[i-1], sma150.iloc[i-1]
    e8, j, s21 = ema8.iloc[i-1], jma.iloc[i-1], sma21.iloc[i-1]
    if any(np.isnan(v) for v in [s50, s150, e8, j, s21]):
        return False, {}

    close_A, open_A = float(df["Close"].iloc[i-1]), float(df["Open"].iloc[i-1])
    close_B, open_B = float(df["Close"].iloc[i]),   float(df["Open"].iloc[i])
    vol_A, vol_B = float(df["Volume"].iloc[i-1]), float(df["Volume"].iloc[i])

    # ── Step 1: structure — SMA50 > SMA150 on the test day ──────────
    structure_ok = s50 > s150

    # ── Step 2: the fast-MA cluster (EMA8/JMA/SMA21) average sits
    #    close to SMA50 on the test day — the fast group is pulling
    #    back to actually touch the medium-term average ─────────────
    cluster_avg = (e8 + j + s21) / 3
    near_pct = abs(cluster_avg - s50) / s50 * 100 if s50 > 0 else 999
    near_ok = near_pct <= cfg["near_sma50_pct"]

    # ── Step 3: candle A red & below the cluster; candle B green &
    #    above the cluster — a clean one-day reclaim of the tested
    #    zone ──────────────────────────────────────────────────────
    candle_A_red   = close_A < open_A and close_A < cluster_avg
    candle_B_green = close_B > open_B and close_B > cluster_avg

    # ── Step 4: volume higher on the reclaim day ─────────────────────
    vol_ok = vol_B > vol_A

    passed = structure_ok and near_ok and candle_A_red and candle_B_green and vol_ok

    return passed, {
        "idx": i, "near_pct": near_pct, "cluster_avg": cluster_avg,
        "sma50": float(s50), "sma150": float(s150),
        "close_A": close_A, "open_A": open_A,
        "close_B": close_B, "open_B": open_B,
        "high_B": float(df["High"].iloc[i]), "low_A": float(df["Low"].iloc[i-1]),
        "vol_A": vol_A, "vol_B": vol_B,
        "structure_ok": structure_ok, "near_ok": near_ok,
        "candle_A_red": candle_A_red, "candle_B_green": candle_B_green,
        "vol_ok": vol_ok,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the fixed 653-ticker scan ──────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_structure": 0,
    "passed_step2_near_sma50": 0,
    "passed_step3_test_and_reclaim": 0,
    "passed_step4_volume": 0,
}

def find_pattern_signals(df, ema8, jma, sma21, sma50, sma150, cfg):
    """
    Scans the last `signal_lookback_days` trading days for the full
    pattern. Returns a list of hit dicts, most recent first. Also
    tallies FUNNEL_COUNTS for every day checked, regardless of match.
    """
    global FUNNEL_COUNTS
    n = len(df)
    lb = cfg["signal_lookback_days"]
    hits = []
    for back in range(0, lb):
        i = (n - 1) - back
        passed, details = check_pattern_at(df, ema8, jma, sma21, sma50, sma150, i, cfg)
        if not details:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if details["structure_ok"]:
            FUNNEL_COUNTS["passed_step1_structure"] += 1
            if details["near_ok"]:
                FUNNEL_COUNTS["passed_step2_near_sma50"] += 1
                if details["candle_A_red"] and details["candle_B_green"]:
                    FUNNEL_COUNTS["passed_step3_test_and_reclaim"] += 1
                    if details["vol_ok"]:
                        FUNNEL_COUNTS["passed_step4_volume"] += 1
        if passed:
            hits.append(details)
    return hits

# ── Technical signal: highlighted pattern (cluster tests SMA50) ────
def analyze_highlighted_pattern(sym, df_daily):
    """
    Returns dict with score and setup details, or None if no
    required condition is met anywhere in the lookback window.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma150_period"] + CFG["signal_lookback_days"] + 20:
        return None

    ema8   = df_daily["Close"].ewm(span=CFG["ema8_period"], adjust=False).mean()
    sma21  = df_daily["Close"].rolling(CFG["sma21_period"]).mean()
    sma50  = df_daily["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df_daily["Close"].rolling(CFG["sma150_period"]).mean()
    jma    = calc_jma(df_daily["Close"], CFG["jma_period"], CFG["jma_phase"])

    hits = find_pattern_signals(df_daily, ema8, jma, sma21, sma50, sma150, CFG)
    if not hits:
        return None

    sig = hits[0]   # most recent
    sig_idx = sig["idx"]
    days_since_signal = (n - 1) - sig_idx
    recent_signals = [
        {"date": df_daily.index[h["idx"]], "bars_ago": (n-1)-h["idx"]}
        for h in hits
    ]

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = sig["high_B"]     # breakout above candle B's high
    stop_loss   = sig["low_A"]      # candle A's low
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
    vol_ratio = sig["vol_B"] / sig["vol_A"] if sig["vol_A"] > 0 else 0

    # ── Match score (0-100) — how well this fits the highlighted
    #    pattern: tightness of the SMA50 test, volume strength,
    #    freshness ─────────────────────────────────────────────────
    score = 0
    reasons = []
    score += max(0, min(40, 40 - sig["near_pct"] * (40/CFG["near_sma50_pct"])))
    reasons.append(f"NearSMA50({sig['near_pct']:.2f}%)")
    score += min(35, (vol_ratio - 1.0) * 35)
    reasons.append(f"Vol{vol_ratio:.1f}x")
    freshness_pts = max(0, 15 - days_since_signal)
    score += freshness_pts
    reasons.append(f"{days_since_signal}dAgo")
    score += 10   # base for clearing every gate
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Near_SMA50_%"   : round(sig["near_pct"], 2),
        "Cluster_Avg"    : round(sig["cluster_avg"], 2),
        "SMA50"          : round(sig["sma50"], 2),
        "SMA150"         : round(sig["sma150"], 2),
        "Vol_Ratio"      : round(vol_ratio, 2),
        "Signal_Date"    : df_daily.index[sig_idx].strftime("%Y-%m-%d"),
        "Days_Since_Signal"  : days_since_signal,
        "Recent_Signal_Count": len(recent_signals),
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}" for s in recent_signals),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_ema8"          : ema8, "_jma": jma, "_sma21": sma21,
        "_sma50"         : sma50, "_sma150": sma150,
    }

# ── Download: daily (→ also Weekly + Monthly via resample) ──────
def _clean(df, min_bars=200):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index,"tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Close","Volume"], inplace=True)
    return df if (len(df)>=min_bars and float(df["Close"].iloc[-1])>0) else None

def download_daily(symbols, days):
    end = datetime.today(); start = end - timedelta(days=days)
    out = {}
    try:
        raw = yf.download(symbols, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), group_by="ticker",
                          auto_adjust=True, actions=False,
                          threads=True, progress=False)
        if raw is not None and not raw.empty:
            pf = {"Open","High","Low","Close","Volume","Adj Close"}
            if isinstance(raw.columns, pd.MultiIndex):
                l0 = set(raw.columns.get_level_values(0))
                for sym in symbols:
                    try:
                        df = raw.xs(sym,axis=1,level=1) if l0&pf else raw[sym]
                        df = _clean(df, min_bars=300)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=300)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=300)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Near_SMA50_%","Vol_Ratio","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Near_SMA50_%":13,"Vol_Ratio":11,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Vol_Ratio":"{:.2f}x"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 3 timeframes")
    print("━"*95)
    h = "".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS)
    print(h)
    print("  " + "─"*93)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val = r.get(c,"—")
        w   = _CW.get(c,12)
        fmt = _CF.get(c)
        try:   s = fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except Exception: s = str(val)
        row += f"  {s:<{w}}"
    print(row)

# ── Health check ──────────────────────────────────────────────
print("━"*65)
print("  STEP 1  DATA CHECK")
print("━"*65)
chk_d = download_daily(["AAPL","MSFT","NVDA"], CFG["history_days"])
if not chk_d:
    print("❌  No data.")
else:
    for s, dd in chk_d.items():
        print(f"  ✅ {s}: daily {len(dd)} bars (${float(dd['Close'].iloc[-1]):.2f}, "
              f"{dd.index[-1].date()})")
print()
# ── Ticker list ───────────────────────────────────────────────
print("━"*65)
print("  STEP 2  FIXED TICKER UNIVERSE")
print("━"*65)
print(f"  Using the fixed {len(FIXED_TICKERS)}-ticker list from the Master")
print(f"  Technical Scanner's 2026-09-10 05:37 UTC run (no full-universe fetch)")

TICKERS = sorted(set(FIXED_TICKERS))
print(f"\n  🎯 Total: {len(TICKERS)} tickers")
print()

# ── Main scan — single pass (daily download only, then check) ────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (single download per ticker)")
print("  A stock only matches if all 3 candles and the structure gate")
print("  fire together within the lookback window\n")

_hdr_done = False
results = []
no_daily_data = 0

daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]

daily_map = {}

with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])

got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers")
print()

for sym in tqdm(list(daily_map.keys()), desc="Checking 3-candle pattern", unit="stk"):
    try:
        r = analyze_highlighted_pattern(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        live_print(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  ✅ Matches       : {len(results)}")
print(f"{'━'*65}")

# ── Diagnostic funnel — where ticker-days were filtered out ──────
print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — where ticker-days were filtered out")
print(f"  (tallied across every liquid ticker's last {CFG['signal_lookback_days']} trading days)")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Ticker-days checked                       : {fc['days_checked']}")
print(f"  Step 1 — structure (SMA50 > SMA150)        : {fc['passed_step1_structure']}")
print(f"  Step 2 — + cluster average near SMA50      : {fc['passed_step2_near_sma50']}")
print(f"  Step 3 — + red test day, green reclaim day : {fc['passed_step3_test_and_reclaim']}")
print(f"  Step 4 — + volume higher on reclaim day    : {fc['passed_step4_volume']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   near_sma50_pct               3.0 → 5.0   (allow a looser SMA50 test)")
    print("   signal_lookback_days           15 → 25    (search further back)")
    print("   min_price                        2 → 1")
    print("   min_avg_volume               80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Near_SMA50_%","Cluster_Avg","SMA50","SMA150","Vol_Ratio",
    "Signal_Date","Days_Since_Signal","Recent_Signal_Count","Recent_Signals",
    "Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"       : lambda v: f"${v:.2f}",
    "Score"       : lambda v: f"{v:.0f}",
    "Entry_Price" : lambda v: f"${v:.2f}",
    "Stop_Loss"   : lambda v: f"${v:.2f}",
    "Risk_%"      : lambda v: f"{v:.1f}%",
    "Near_SMA50_%": lambda v: f"{v:.2f}%",
    "Cluster_Avg": lambda v: f"${v:.2f}",
    "JMA"         : lambda v: f"${v:.2f}",
    "EMA8"        : lambda v: f"${v:.2f}",
    "SMA50"       : lambda v: f"${v:.2f}",
    "SMA150"      : lambda v: f"${v:.2f}",
    "Vol_Ratio"   : lambda v: f"{v:.2f}x",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
            "Near_SMA50_%","Vol_Ratio","Signal_Date"]
    DISP = [c for c in DISP if c in df_out.columns]

    gc = "#22c55e"

    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;text-align:center;'
        f'border-bottom:2px solid {gc};white-space:nowrap">{c}</th>'
        for c in DISP
    )
    rows_html = ""
    for i, r in enumerate(results):
        bg  = "#ffffff" if i%2==0 else "#f0f9ff"
        tds = ""
        for col in DISP:
            raw  = r.get(col)
            disp = fmt_v(col, raw)
            sty  = ""
            if col == "Score":
                try:
                    v = float(raw)
                    g = int(min(220, 80 + v*1.4))
                    sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except Exception: pass
            elif col == "Near_SMA50_%":
                try:
                    v = float(str(raw).replace("%",""))
                    clr = "#22c55e" if v <= 1 else "#f59e0b"
                    sty = f"color:{clr};font-weight:600"
                except Exception: pass
            tds += (f'<td style="padding:7px 12px;font-size:12px;'
                    f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                    f'{disp}</td>')
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

    table_html = f"""
<div style="margin:14px 0">
  <div style="background:linear-gradient(90deg,{gc}22,#0f172a);
          border-left:4px solid {gc};border-radius:6px 6px 0 0;
          padding:10px 18px;display:flex;align-items:center;gap:10px">
    <span style="font-size:18px">🎯</span>
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Highlighted Pattern (Cluster Tests SMA50)</span>
    <span style="color:{gc};font-size:12px;margin-left:8px">{len(results)} stock{'s' if len(results)!=1 else ''}</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-top:none;
          border-radius:0 0 8px 8px;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
    <table style="border-collapse:collapse;width:100%;min-width:700px">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    header_html = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Highlighted Pattern (Cluster Tests SMA50)
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; pattern checked over the last {CFG['signal_lookback_days']} trading days
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Price above SMA50, SMA150, AND JMA &nbsp;·&nbsp;
  EMA8 above SMA50 AND SMA150 &nbsp;·&nbsp;
  JMA still above EMA8 but the gap is tight (coming close from above) &nbsp;·&nbsp;
  Price closed above both JMA and EMA8 today &nbsp;·&nbsp;
  Volume higher than the previous day &nbsp;·&nbsp;
  Entry_Price = the breakout day's high &nbsp;·&nbsp;
  Stop_Loss = the breakout day's low — reasonable defaults, not explicitly
  requested &nbsp;·&nbsp;
  already holding above both &nbsp;·&nbsp;
  Signal_Date is the exact date the pattern fired (checked over the
  last {CFG['signal_lookback_days']} trading days, not just today)
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "Near_SMA50_%","Vol_Ratio","Signal_Date"]
    CLI_COLS = [c for c in CLI_COLS if c in df_out.columns]
    col_w = {c: max(len(c), max(
        len(fmt_v(c, df_out[c].iloc[i])) for i in range(len(df_out))
    ))+2 for c in CLI_COLS}
    top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
    sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
    bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
    hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
    inner= sum(col_w.values()) + len(CLI_COLS) - 1
    print()
    print(f"  ╔{'═'*inner}╗")
    tit = f"  Highlighted Pattern (653-ticker scan)   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
    print(f"  ║{tit.center(inner)}║")
    print(f"  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐")
    print(f"  │{hdr}│")
    print(f"  ├{sep}┤")
    for i,(_, row_) in enumerate(df_out.iterrows()):
        cells=[fmt_v(c,row_.get(c)).center(col_w[c]) for c in CLI_COLS]
        print(f"  │{'│'.join(cells)}│")
        if i<len(df_out)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")
    print(f"""
  COLUMN KEY
  ──────────────────────────────────────────────────────
  Score           0-100 (convergence tightness + volume increase +
                  freshness)
  Entry_Price     the breakout day's high (a reasonable default,
                  not explicitly requested)
  Stop_Loss       the breakout day's low
  Near_SMA50_%    how close the fast-MA cluster average was to
                  pre-breakout day
  JMA / EMA8 / SMA50 / SMA150  their values on the breakout day
  Vol_Ratio       breakout day volume / previous day volume
  Signal_Date     exact calendar date the pattern fired
  Days_Since_Signal  how many trading days ago (0 = today; checked
                     over the last signal_lookback_days trading days)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"highlighted_pattern_653_scan_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_highlighted_pattern_653_scan_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Highlighted Pattern 653-Ticker Scan {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                      "Near_SMA50_%","Vol_Ratio","Signal_Date"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            near   = r.get("Near_SMA50_%",0) or 0
            vr     = r.get("Vol_Ratio",0) or 0
            sdate  = r.get("Signal_Date","—")
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{float(conv):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(vr):.2f}x</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{sdate}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="8" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Highlighted Pattern — 653-Ticker Scan
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — JMA converging toward
  EMA8 from above, then price breaking above both on rising volume
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — where ticker-days were filtered out (last {CFG['signal_lookback_days']} trading days, all 653 tickers)
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {FUNNEL_COUNTS['days_checked']} ticker-days checked &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step1_structure']} passed Step 1 (structure) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step2_near_sma50']} passed Step 2 (cluster near SMA50) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step3_test_and_reclaim']} passed Step 3 (red test, green reclaim) &nbsp;→&nbsp;
    <b style="color:#facc15">{FUNNEL_COUNTS['passed_step4_volume']} passed Step 4 (volume) = full match</b>
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:600px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">
  📎 Full results attached as CSV
</p>
{f'''<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:6px;padding:10px 14px">
  <p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">
    TICKERS (comma-separated, copy/paste)
  </p>
  <p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;
      word-break:break-all">{", ".join(r.get("Ticker","") for r in rl)}</p>
</div>''' if rl else ''}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Highlighted Pattern (Cluster Tests SMA50), 653-ticker scan — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (fast-MA cluster tests SMA50 + red/green reclaim + volume up)",
            "="*60,
            f"FUNNEL: {FUNNEL_COUNTS['days_checked']} ticker-days -> "
            f"{FUNNEL_COUNTS['passed_step1_structure']} passed structure -> "
            f"{FUNNEL_COUNTS['passed_step2_near_sma50']} passed near-SMA50 -> "
            f"{FUNNEL_COUNTS['passed_step3_test_and_reclaim']} passed test+reclaim -> "
            f"{FUNNEL_COUNTS['passed_step4_volume']} passed volume (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                near   = r.get("Near_SMA50_%",0) or 0
                vr     = r.get("Vol_Ratio",0) or 0
                sdate  = r.get("Signal_Date","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"Convergence:{float(conv):.2f}%  Vol:{float(vr):.2f}x  Signal:{sdate}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Highlighted Pattern (653 tickers) — {cnt} match{'es' if cnt!=1 else ''}"
                f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj
        msg["From"]    = gu
        msg["To"]      = ", ".join(eto)

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e, "plain"))
        alt.attach(MIMEText(html_e,  "html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
            sz = os.path.getsize(csv_path)
            print(f"[Email] 📎 Attached: {os.path.basename(csv_path)} ({sz:,} bytes)")
        except Exception as e:
            print(f"[Email] ⚠️  CSV attach failed: {e}")

    try:
        print(f"[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent successfully to: {', '.join(eto)}")
        print(f"[Email]    Subject: {subj}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTHENTICATION FAILED")
        print("         GMAIL_PASS must be a Gmail App Password, NOT your login password")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV is still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5 (price + JMA/EMA8/SMA21/SMA50/SMA150) ──────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,4.2*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]
    for ax, r in zip(axes, top):
        df_p = r["_df_daily"].tail(90).copy()
        jma_p   = r["_jma"].reindex(df_p.index)
        ema8_p  = r["_ema8"].reindex(df_p.index)
        sma21_p = r["_sma21"].reindex(df_p.index)
        sma50_p = r["_sma50"].reindex(df_p.index)
        sma150_p= r["_sma150"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.4, label="Close", zorder=5)
        ax.plot(df_p.index, jma_p,   color="#f472b6", lw=1.0, label="JMA",   zorder=4)
        ax.plot(df_p.index, ema8_p,  color="#38bdf8", lw=1.0, ls="--", label="EMA8",  zorder=4)
        ax.plot(df_p.index, sma21_p, color="#34d399", lw=1.0, ls="--", label="SMA21", zorder=4)
        ax.plot(df_p.index, sma50_p, color="#fbbf24", lw=1.2, ls="-.", label="SMA50", zorder=3)
        ax.plot(df_p.index, sma150_p,color="#f87171", lw=1.1, ls=":",  label="SMA150",zorder=3)

        sig_date = pd.to_datetime(r["Signal_Date"])
        if sig_date in df_p.index:
            ax.scatter([sig_date], [df_p.loc[sig_date,"Close"]], color="#22c55e", s=60,
                       zorder=6, marker="^", label="Breakout")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.8,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"Near {r['Near_SMA50_%']:.2f}%  |  Vol {r['Vol_Ratio']:.2f}x  |  {r['Signal_Date']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Highlighted Pattern — 653-Ticker Scan  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"highlighted_pattern_653_scan_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required — scanned over the last
  signal_lookback_days trading days, {CFG['signal_lookback_days']}d ≈ 3 weeks,
  not just today)

  1) STRUCTURE (on the test day): SMA50 > SMA150.
  2) CLUSTER NEAR SMA50: the average of the fast-MA cluster
     (EMA8, JMA, SMA21) sits within near_sma50_pct% of SMA50 on
     the test day — the fast group is pulling back to actually
     touch/test the medium-term average, matching the highlighted
     chart's compression zone.
  3) TEST + RECLAIM: the test day is RED and closed BELOW the
     cluster average; the next day is GREEN and closed ABOVE the
     cluster average — a clean one-day reclaim of the tested zone.
  4) VOLUME: the reclaim day's volume is higher than the test day's.

  If the pattern fired more than once in the window, the MOST
  RECENT occurrence is used for Entry/Stop/scoring.

  📋 UNIVERSE
  Restricted to EXACTLY the 653 tickers from the Master Technical
  Scanner's 2026-09-10 05:37 UTC run (hardcoded list, no full-
  universe fetch) — per the request to check "those 653 tickers."

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~400 days).

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = the reclaim day's HIGH
  Stop_Loss   = the test day's LOW
  Signal_Date = the exact calendar date the pattern fired
  Days_Since_Signal = how many trading days ago (0 = today)
  Recent_Signals    = every date the pattern fired within the window

  📋 SCORE (0-100)
  Tightness of the SMA50 test (0-40) + volume increase strength
  (0-35) + freshness (0-15) + base points for clearing every gate (10)

  💡 BEST SETUPS
  Score > 70                tight test, strong volume, fresh
  Near_SMA50_% < 1%             the cluster barely touched SMA50
  Vol_Ratio > 1.5                  well above the test day's volume
  Days_Since_Signal = 0-3             freshest signal

  ⚙️  TUNE IF 0 RESULTS
  near_sma50_pct                3.0 → 5.0   (allow a looser SMA50 test)
  signal_lookback_days            15 → 25    (search further back)
  min_price                         2 → 1
  min_avg_volume                80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
