"""Diagnose why the conditional var_scale lever degrades the 50-70% market
midband in EVERY grid cell of tune_cond_var_scale_2223.py, even the mildest
(threshold=0.95, scale=0.7). The condition triggers on the MODEL's own
point-estimate extremity, not the market's, so a midband (by market) row can
still get its variance shrunk if the model itself disagrees with the market
and calls the row lopsided. Quantify how many midband rows get touched at
each threshold and whether that set is disproportionately wrong."""
import numpy as np
import pandas as pd

SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\cond_tune_window_2223.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df["fav_extremity"] = np.maximum(df.p_winner_hier, 1 - df.p_winner_hier)
    mid = df[(df.mkt_fav > 0.5) & (df.mkt_fav <= 0.7)].copy()
    print(f"midband (market 50-70%) n={len(mid)}")
    for threshold in [0.80, 0.85, 0.90, 0.95, 0.98]:
        touched = mid[mid.fav_extremity > threshold]
        n_touched = len(touched)
        if n_touched:
            # is the model's favourite (fav_extremity side) actually correct more/less often
            # than baseline accuracy in this touched subset?
            model_fav_is_winner = (touched.p_winner_hier > 0.5).astype(int) == 1  # winner is label 1 always? need actual label
            print(f"  threshold={threshold}: n_touched={n_touched} ({100*n_touched/len(mid):.1f}% of midband)")
        else:
            print(f"  threshold={threshold}: n_touched=0")

    # also: overall grid n_touched (all rows, not just midband) for context
    print("\noverall (all tuning-window rows) touch counts:")
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        n_touched = int((df.fav_extremity > threshold).sum())
        print(f"  threshold={threshold}: n_touched={n_touched} ({100*n_touched/len(df):.1f}% of all rows)")

    # log-loss of the touched-midband subset at baseline (p_winner_hier_mc, var_scale=1) vs the model's raw point estimate
    print("\nmidband-touched-at-0.85 subset: baseline mc-averaged log-loss vs raw point-estimate log-loss")
    touched85 = mid[mid.fav_extremity > 0.85]
    if len(touched85):
        def ll(p):
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return float(-np.log(p).mean())
        print(f"  n={len(touched85)}")
        print(f"  mc600 baseline ll (need p_winner_hier_mc600 col? checking columns)")
        print(list(df.columns))
