"""Pete's rust idea (2026-09-09): a decayed count of recent matches, so a
returning player is still 'rusty' in his second and third match back rather
than fully recovered the moment weeks_since_last drops to 0. Candidate:
    recent_load = sum over prior matches of exp(-days_ago / TAU)
Printed for Alcaraz's and Zverev's 2026 US Open run and for a regular tour
week, at three TAU values, plus the field distribution."""
import numpy as np, pandas as pd
from pathlib import Path
S = Path("C:/dev/dropshotverse/dropshotdata/tune_scratch")
e = pd.read_parquet(S/"elo_raw_ratings_full.parquet")[["date","tour","level_file","winner_uid","loser_uid","winner_name","loser_name","round"]]
e = e[e.tour=="atp"]
long = pd.concat([e[["date","winner_uid","winner_name","round","level_file"]].set_axis(["date","uid","name","round","lvl"],axis=1),
                  e[["date","loser_uid","loser_name","round","level_file"]].set_axis(["date","uid","name","round","lvl"],axis=1)]).sort_values("date")
def load_before(uid, d, tau):
    h = long[(long.uid==uid)&(long.date<d)]
    return float(np.exp(-(d-h.date).dt.days/tau).sum())
US = pd.Timestamp("2026-08-30")
for nm, uid in [("Alcaraz","atp:207989"),("Zverev","atp:100644")]:
    print(f"\n=== {nm}: matches from 2026-07-25, with recent_load BEFORE each match (tau = 14 / 21 / 35 days) ===")
    h = long[(long.uid==uid)&(long.date>="2026-07-25")]
    rows = []
    for d, rd, lv in zip(h.date, h["round"], h.lvl):
        rows.append({"date": d.date(), "round": rd, "level": lv, **{f"tau{t}": round(load_before(uid, d, t),2) for t in (14,21,35)}})
    # plus the US Open matches he'd have played: R128..QF on 30 Aug, 1, 3, 5 Sep
    for k, d in enumerate(pd.to_datetime(["2026-08-31","2026-09-02","2026-09-04","2026-09-06"])):
        fake = long[(long.uid==uid)&(long.date<d)]
        extra = [pd.Timestamp("2026-08-31")+pd.Timedelta(days=2*i) for i in range(k)]
        vals = {f"tau{t}": round(float(np.exp(-(d-fake.date).dt.days/t).sum() + sum(np.exp(-(d-x).days/t) for x in extra)),2) for t in (14,21,35)}
        rows.append({"date": d.date(), "round": f"USO match {k+1}", "level": "(hypothetical)", **vals})
    print(pd.DataFrame(rows).to_string(index=False))
# field distribution at the US Open start
field = long[(long.date>=US-pd.Timedelta(days=120))&(long.date<US)].uid.unique()
vals = pd.Series({u: load_before(u, US, 21) for u in field})
print("\nrecent_load (tau=21) over ATP players active in the 120 days before 2026-08-30, n=%d:" % len(vals))
print(vals.describe().round(2).to_string())
