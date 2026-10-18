# Builds the JMA-cross email section from the MA-support one (same layout,
# different columns / wording). Used by claude_run/tools/assemble_jma_cross.py.
def build_email_section(src):
    def rep(a, b):
        nonlocal src
        assert src.count(a) == 1, ("expected once", a[:60])
        src = src.replace(a, b)
    rep('''        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","Support",
                 "MA","vs MA","Bounces","Held","Qty","Amount"]''',
'''        heads = ["Ticker","Price","Score","Entry","Stop (cross)","Risk %","Cross",
                 "Confirm","Close vs JMA","Vol/20d","Trend","Qty","Amount"]''')
    rep('''                + td(r.get("Support_MA","—"), "font-weight:700;color:#7c3aed")
                + td(f'${float(r.get("MA_Value",0) or 0):.2f}')
                + td(f'{float(r.get("Dist_to_MA_%",0) or 0):+.1f}%')
                + td(f'{r.get("Bounces","—")}')
                + td(f'{float(r.get("Held_%",0) or 0):.0f}%')''',
'''                + td(r.get("Cross_Date","—"), "color:#7c3aed")
                + td(r.get("Confirm_Date","—"), "color:#a78bfa;font-weight:600")
                + td(f'{float(r.get("Close_vs_JMA_%",0) or 0):+.1f}%')
                + td(f'x{float(r.get("Vol_vs_20d_Avg",0) or 0):.2f}')
                + td("Above SMA50" if r.get("Above_SMA50") else "Below SMA50",
                     "color:#16a34a" if r.get("Above_SMA50") else "color:#b45309")''')
    rep('''  📊 Daily MA Support — EMA21 / SMA50''', '''  📊 Daily JMA Cross Above EMA8''')
    rep('''  {cnt} match{'es' if cnt!=1 else ''} — price took support on the same MA (EMA{CFG['ema_period']} or
  SMA{CFG['sma_period']}) most of the last {CFG['support_window_days']} days and is back near it today''',
'''  {cnt} match{'es' if cnt!=1 else ''} — JMA({CFG['jma_period']}) crossed above EMA{CFG['ema_period']} from below and the
  next day closed above JMA; stop = the price where JMA and EMA{CFG['ema_period']} intersect''')
    rep('''    {fc['passed_step1_support_found']} support MA with &ge;{CFG['min_bounces']} bounces &nbsp;→&nbsp;
    {fc['passed_step2_held_most_of_time']} held &ge;{CFG['min_held_ratio']*100:.0f}% &nbsp;→&nbsp;
    {fc['passed_step3_rising_ma']} MA rising &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_near_ma_now']} near the MA today = full match</b>''',
'''    {fc['passed_step1_jma_crossed_above_ema8']} JMA crossed above EMA{CFG['ema_period']} &nbsp;→&nbsp;
    {fc['passed_step2_next_close_above_jma']} next close above JMA &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step3_close_above_cross_price']} close above cross price = full match</b>''')
    rep('''Entry = today's close, Stop = {CFG['stop_buffer_pct']:.0f}% below the support MA (defaults)''',
        '''Entry = confirm-day close (default), Stop = JMA/EMA{CFG['ema_period']} cross price''')
    rep('''            f"Daily MA Support (EMA21/SMA50) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (support on the same MA most of the last {CFG['support_window_days']} days; near it today)",''',
'''            f"Daily JMA Cross Above EMA{CFG['ema_period']} — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (JMA crossed above EMA{CFG['ema_period']}; next day closed above JMA; stop = cross price)",''')
    rep('''            f"{fc['passed_step1_support_found']} support MA >={CFG['min_bounces']} bounces -> "
            f"{fc['passed_step2_held_most_of_time']} held >={CFG['min_held_ratio']*100:.0f}% -> "
            f"{fc['passed_step3_rising_ma']} MA rising -> "
            f"{fc['passed_step4_near_ma_now']} near MA today (=full match)",''',
'''            f"{fc['passed_step1_jma_crossed_above_ema8']} JMA crossed above EMA{CFG['ema_period']} -> "
            f"{fc['passed_step2_next_close_above_jma']} next close above JMA -> "
            f"{fc['passed_step3_close_above_cross_price']} close above cross price (=full match)",''')
    rep('''                    f"{r.get('Support_MA','—')}:${float(r.get('MA_Value',0) or 0):.2f} "
                    f"({float(r.get('Dist_to_MA_%',0) or 0):+.1f}%)  "
                    f"Bounces:{r.get('Bounces','—')}  Held:{float(r.get('Held_%',0) or 0):.0f}%  "''',
'''                    f"Cross:{r.get('Cross_Date','—')}  Confirm:{r.get('Confirm_Date','—')}  "
                    f"vsJMA:{float(r.get('Close_vs_JMA_%',0) or 0):+.1f}%  "''')
    rep('''        subj = (f"📊 MA Support EMA21/SMA50 — {cnt} signal{'s' if cnt!=1 else ''}"''',
        '''        subj = (f"📊 JMA Cross Above EMA8 — {cnt} signal{'s' if cnt!=1 else ''}"''')
    return src[:src.index("# ── Charts for top 5")]
