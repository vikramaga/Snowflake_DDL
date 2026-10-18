# ── Charts for top 5: price with JMA and EMA8, the cross point (= stop)
#    and the confirm day marked ─────────────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 3.6*len(top)), facecolor="#0f172a",
                              squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax = axes[p]
        df_all = r["_df_daily"]
        df_p = df_all.tail(70)
        idx  = df_p.index
        k, x, t = r["_idx"], r["_cross_idx"], r["_cross_t"]

        ax.set_facecolor("#0f172a")
        ax.plot(idx, df_p["Close"], color="#60a5fa", lw=1.4, label="Close", zorder=5)
        ax.plot(idx, r["_jma"].reindex(idx), color="#f472b6", lw=1.8, label="JMA 13")
        ax.plot(idx, r["_ema"].reindex(idx), color="#22c55e", lw=1.2, label="EMA8")
        ax.plot(idx, r["_sma"].reindex(idx), color="#fbbf24", lw=0.9, ls=":", label="SMA50")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                   label=f"Stop {r['Stop_Loss']:.2f} (cross)")
        d0, d1 = df_all.index[x-1], df_all.index[x]
        cross_d = d0 + (d1 - d0) * t
        if d0 in idx:
            ax.scatter([cross_d], [r["Cross_Price"]], color="#ef4444", marker="x", s=60,
                       zorder=8, label="Cross point")
        sig_d = df_all.index[k]
        if sig_d in idx:
            ax.scatter([sig_d], [df_all["Close"].iloc[k]], color="#facc15", s=50, zorder=7,
                       label="Confirm close")
        ax.set_title(f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  "
                     f"cross {r['Cross_Date']}  |  confirm {r['Confirm_Date']}  |  "
                     f"stop {r['Stop_Loss']:.2f} ({r['Risk_%']:.1f}%)",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=7)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily JMA Cross Above EMA8  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_jma_cross_ema8_chart_{ts}.png")
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
  📋 SIGNAL (all required; confirm day = latest daily candle)
  JMA = JMA({CFG['jma_period']}, phase {CFG['jma_phase']}, power {CFG['jma_power']}); EMA{CFG['ema_period']} on daily closes.

  1) Cross day: JMA was at or below EMA{CFG['ema_period']} the day before and is
     above it on the cross day.
  2) Confirm day (next day): the close is above JMA.
  3) The confirm-day close is above the cross price (valid stop).

  CROSS PRICE = where JMA and EMA{CFG['ema_period']} intersect, interpolated between the
  day before the cross and the cross day. It is the STOP LOSS.

  📋 OUTPUT
  Stop_Loss   = cross price (requested)
  Entry_Price = confirm-day CLOSE (reasonable default)
  Quantity / Total_Amount = sized for a $40 max loss (position_sizing.py)
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 30 + JMA above EMA8 gap (0-15) + close 0-3% over JMA (15; 8 if
  further; -5 if > 6%) + tight stop (0-15) + volume vs 20d avg (0-15)
  + above SMA50 (10); -10 if stop risk > 12%

  ⚙️  TUNE IF 0 RESULTS
  signal_lookback_days            1 → 3
  min_price                      10 → 5
  min_avg_volume             100000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
