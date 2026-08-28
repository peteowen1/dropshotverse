"""Issue 20: validate mc_parallel.analytic_correction() against a
high-precision MC reference (mc_average(n_samples=2000)) before trusting it
as a candidate. Runs on a sample of hier_roundorder_tune_predictions.parquet
(2022-23 tuning-window fit, current v25+round-order-fix config) - stratified
by combined posterior variance so the high-variance tail (where the
second-order Taylor approximation is expected to break down) is actually
represented, not just the bulk of the distribution.

Also times analytic_correction() against mc_average() at both a
production sample count (60, what the harness normally uses) and the
high-precision reference count (2000), for the secondary speed-win report.
"""
import sys
import time
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from mc_parallel import mc_average, analytic_correction

PREDS = r"C:\dev\dropshotverse\dropshotdata\hier_roundorder_tune_predictions.parquet"


def main():
    df = pd.read_parquet(PREDS)
    df = df.assign(combined_var=df["d1_var"] + df["d2_var"])
    df["var_q"] = pd.qcut(df["combined_var"], 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"])

    rng = np.random.default_rng(11)
    n_per_q = 125
    parts = []
    for q, g in df.groupby("var_q", observed=True):
        n = min(n_per_q, len(g))
        parts.append(g.sample(n=n, random_state=rng.integers(0, 1 << 30)))
    sample = pd.concat(parts, ignore_index=True)
    print(f"validation sample: {len(sample)} rows, stratified by combined_var quartile "
         f"({n_per_q}/quartile target)")
    print(sample.groupby("var_q", observed=True)["combined_var"]
         .agg(["count", "min", "max"]))

    print("\n=== timing ===")
    t0 = time.time()
    ref = mc_average(sample, n_samples=2000, seed_base=123)
    t_ref = time.time() - t0
    print(f"mc_average(n_samples=2000, high-precision reference): "
         f"{len(sample)} rows in {t_ref:.2f}s ({t_ref/len(sample)*1000:.2f}ms/row)")

    t0 = time.time()
    prod60 = mc_average(sample, n_samples=60, seed_base=0)
    t_prod = time.time() - t0
    print(f"mc_average(n_samples=60, production default): "
         f"{len(sample)} rows in {t_prod:.2f}s ({t_prod/len(sample)*1000:.2f}ms/row)")

    t0 = time.time()
    an = analytic_correction(sample, h=0.01)
    t_an = time.time() - t0
    print(f"analytic_correction(h=0.01): "
         f"{len(sample)} rows in {t_an:.2f}s ({t_an/len(sample)*1000:.2f}ms/row)")
    print(f"speedup vs mc(n=60):   {t_prod/t_an:.1f}x")
    print(f"speedup vs mc(n=2000): {t_ref/t_an:.1f}x")

    sample = sample.assign(mc_ref=ref, mc_prod60=prod60, analytic=an)
    sample["diff_an_ref"] = sample["analytic"] - sample["mc_ref"]
    sample["diff_prod60_ref"] = sample["mc_prod60"] - sample["mc_ref"]

    print("\n=== overall agreement: analytic vs high-precision MC(n=2000) ===")
    print(f"mean abs diff:    {sample['diff_an_ref'].abs().mean():.5f}")
    print(f"median abs diff:  {sample['diff_an_ref'].abs().median():.5f}")
    print(f"max abs diff:     {sample['diff_an_ref'].abs().max():.5f}")
    print(f"mean signed diff: {sample['diff_an_ref'].mean():+.5f} "
         f"(sign shows systematic direction, if any)")
    print(f"correlation (analytic, mc_ref): "
         f"{np.corrcoef(sample['analytic'], sample['mc_ref'])[0,1]:.5f}")
    print(f"\n(for context) mc(n=60, the harness's normal sample count) vs "
         f"mc(n=2000) MC-noise-only disagreement:")
    print(f"mean abs diff:    {sample['diff_prod60_ref'].abs().mean():.5f}")
    print(f"max abs diff:     {sample['diff_prod60_ref'].abs().max():.5f}")

    print("\n=== agreement broken down by combined-variance quartile ===")
    g = sample.groupby("var_q", observed=True)
    tbl = g.apply(lambda d: pd.Series({
        "n": len(d),
        "mean_combined_var": d["combined_var"].mean(),
        "mean_abs_diff_analytic": d["diff_an_ref"].abs().mean(),
        "max_abs_diff_analytic": d["diff_an_ref"].abs().max(),
        "mean_signed_diff_analytic": d["diff_an_ref"].mean(),
        "mean_abs_diff_mc60_vs_mc2000": d["diff_prod60_ref"].abs().mean(),
    }), include_groups=False)
    print(tbl.to_string())

    print("\n=== agreement broken down by point-estimate confidence bucket ===")
    sample["conf"] = np.maximum(sample["p_winner_hier"], 1 - sample["p_winner_hier"])
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample["conf_bucket"] = pd.cut(sample["conf"], bins)
    g2 = sample.groupby("conf_bucket", observed=True)
    tbl2 = g2.apply(lambda d: pd.Series({
        "n": len(d),
        "mean_combined_var": d["combined_var"].mean(),
        "mean_abs_diff_analytic": d["diff_an_ref"].abs().mean(),
        "max_abs_diff_analytic": d["diff_an_ref"].abs().max(),
        "mean_signed_diff_analytic": d["diff_an_ref"].mean(),
    }), include_groups=False)
    print(tbl2.to_string())

    # Flag the worst individual disagreements for inspection - are they
    # explainable (high variance) or something else (a bug)?
    print("\n=== 10 worst individual disagreements (analytic vs mc_ref) ===")
    worst = sample.reindex(sample["diff_an_ref"].abs().sort_values(ascending=False).index).head(10)
    print(worst[["d1_mean", "d1_var", "d2_mean", "d2_var", "best_of",
                "p_winner_hier", "mc_ref", "analytic", "diff_an_ref"]].to_string())

    sample.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\analytic_validate_sample.parquet")
    print("\nwrote tune_scratch/analytic_validate_sample.parquet")


if __name__ == "__main__":
    main()
