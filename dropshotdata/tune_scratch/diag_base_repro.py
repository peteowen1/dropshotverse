"""Diagnostic (read-only): reproduce the base Elo/Hier/Market comparison on the
2024-25 holdout, exactly per the join pattern given in the task brief. Does not
modify any modeling script. Writes nothing except printed output (and, if
IS_MAIN, a cached joined parquet under tune_scratch/diag_joined.parquet for the
other diag_* scripts to reuse without repaying the odds-join cost).
"""
import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from market_compare import key_from_full, key_from_short, devig_power, load_odds

HIER_PATH = r"C:\dev\dropshotverse\dropshotdata\hier_v25_eval_predictions.parquet"


def build_joined(hier_path=HIER_PATH):
    elo = pd.read_parquet(r'C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet')
    elo = elo[elo.level_file == 'tour'][['date', 'tour', 'winner_uid', 'loser_uid',
                                          'winner_name', 'loser_name', 'p_winner']].rename(
        columns={'p_winner': 'p_elo'})
    hier = pd.read_parquet(hier_path)
    m = elo.merge(hier, on=['date', 'tour', 'winner_uid', 'loser_uid'], how='inner', suffixes=('', '_h'))
    m['k_w'] = m['winner_name'].map(key_from_full)
    m['k_l'] = m['loser_name'].map(key_from_full)
    m['pair'] = m.apply(lambda r: '|'.join(sorted([r.k_w, r.k_l])), axis=1)
    odds = load_odds(years=(2024, 2025)).dropna(subset=['Winner', 'Loser', 'Date'])
    odds['k_w'] = odds['Winner'].map(key_from_short)
    odds['k_l'] = odds['Loser'].map(key_from_short)
    odds['date'] = pd.to_datetime(odds['Date'], errors='coerce')
    odds = odds.dropna(subset=['date'])
    odds['pair'] = odds.apply(lambda r: '|'.join(sorted([r.k_w, r.k_l])), axis=1)
    j = m.merge(odds[['pair', 'tour', 'date', 'k_w', 'k_l', 'PSW', 'PSL']],
                on=['pair', 'tour'], how='left', suffixes=('', '_o'))
    j['gap'] = (j['date_o'] - j['date']).dt.days
    j = j[(j['gap'] >= -3) & (j['gap'] <= 21)]
    j = j.sort_values('gap').drop_duplicates(subset=['pair', 'tour', 'date'], keep='first')
    j = j[(j['k_w_o'] == j['k_w']) & j[['PSW', 'PSL']].notna().all(axis=1)]
    j['p_mkt'] = devig_power(j['PSW'].values.astype(float), j['PSL'].values.astype(float))
    j = j.dropna(subset=['p_mkt'])
    return j


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


def acc(p):
    return float((np.asarray(p, dtype=float) > 0.5).mean())


if __name__ == "__main__":
    j = build_joined()
    print(f"n rows joined: {len(j):,}")
    print(f"hier parquet used: {HIER_PATH}")
    for name, col in [("Elo", "p_elo"), ("Hier (raw p_winner_hier)", "p_winner_hier"),
                       ("Hier (MC, p_winner_hier_mc)", "p_winner_hier_mc"),
                       ("Market", "p_mkt")]:
        print(f"  {name:32s} n={j[col].notna().sum():6d}  log-loss={ll(j[col]):.4f}  acc={acc(j[col]):.4f}")
    j.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_joined.parquet")
    print("wrote tune_scratch/diag_joined.parquet")
