# ============================================================
# Position sizing shared by all scanners.
#
# Adds two columns to a results DataFrame, right after Stop_Loss:
#   Quantity      = floor(MAX_LOSS / (Entry - Stop_Loss))   whole shares
#   Total_Amount  = Quantity * Entry                        cost to buy
#
# so that if the stop is hit, the loss is at most MAX_LOSS.
# Quantity is 0 when one share alone would risk more than MAX_LOSS,
# or when the stop is missing / not below the entry.
#
# Change the risk per trade in ONE place: MAX_LOSS_USD below (or set
# the MAX_LOSS_USD environment variable).
#
# Entry column: Entry_Price if present, otherwise Price.
# Stop column : Stop_Loss.
# For rupee-priced (NSE) scanners pass currency="INR": the USD limit
# is converted at the live USD/INR rate (fallback 88 if unavailable).
# ============================================================
import math
import os

MAX_LOSS_USD = float(os.environ.get("MAX_LOSS_USD", "40"))
_FALLBACK_USDINR = 88.0


def _usd_to(currency):
    if currency.upper() == "USD":
        return 1.0, "fixed"
    if currency.upper() == "INR":
        try:
            import yfinance as yf
            px = yf.Ticker("INR=X").history(period="5d")["Close"].dropna()
            if len(px) and 50 < float(px.iloc[-1]) < 150:
                return float(px.iloc[-1]), "live"
        except Exception:
            pass
        return _FALLBACK_USDINR, "fallback"
    raise ValueError(f"unsupported currency {currency}")


def size_position(entry, stop, max_loss):
    """Whole-share quantity whose loss at the stop is <= max_loss."""
    try:
        entry, stop = float(entry), float(stop)
    except (TypeError, ValueError):
        return 0, 0.0
    if not (math.isfinite(entry) and math.isfinite(stop)) or entry <= 0:
        return 0, 0.0
    risk_ps = entry - stop
    if risk_ps <= 0:
        return 0, 0.0
    qty = int(math.floor(max_loss / risk_ps + 1e-9))
    return qty, round(qty * entry, 2)


def add_position_size(df, max_loss_usd=None, currency="USD",
                      entry_col=None, stop_col="Stop_Loss"):
    """Return a copy of df with Quantity and Total_Amount added after the stop column."""
    max_loss_usd = MAX_LOSS_USD if max_loss_usd is None else float(max_loss_usd)
    rate, src = _usd_to(currency)
    max_loss = max_loss_usd * rate

    out = df.copy()
    if entry_col is None:
        entry_col = "Entry_Price" if "Entry_Price" in out.columns else "Price"
    if entry_col not in out.columns or stop_col not in out.columns:
        print(f"  ⚠️  Position sizing skipped: need '{entry_col}' and '{stop_col}' columns")
        return out

    sized = [size_position(e, s, max_loss) for e, s in zip(out[entry_col], out[stop_col])]
    qty   = [q for q, _ in sized]
    total = [t for _, t in sized]
    for col in ("Quantity", "Total_Amount"):
        if col in out.columns:
            out = out.drop(columns=col)
    pos = list(out.columns).index(stop_col) + 1
    out.insert(pos, "Quantity", qty)
    out.insert(pos + 1, "Total_Amount", total)

    note = (f" (= {max_loss:,.0f} {currency.upper()} at USD/INR {rate:.2f}, {src})"
            if currency.upper() != "USD" else "")
    n_zero = sum(1 for q in qty if q == 0)
    print(f"  💰 Position sizing: max loss ${max_loss_usd:,.0f}{note} per trade"
          f" — {len(out) - n_zero} sized, {n_zero} with Quantity 0 (stop wider than the max loss per share, or no valid stop)")
    return out
