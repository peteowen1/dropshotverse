"""Reusable: join elo_eval_predictions.parquet (tour level) to market odds and
score - the exact pattern from RESUME-beat-elo.md step 4, reused as-is so the
"gap to Elo" comparison point stays apples-to-apples across rounds."""
import numpy as np
import pandas as pd

from market_compare import key_from_full, key_from_short, devig_power, load_odds

elo = pd.read_parquet(r'C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet')
elo = elo[elo.level_file == 'tour'][
    ['date', 'tour', 'winner_uid', 'loser_uid', 'winner_name', 'loser_name', 'p_winner']
].rename(columns={'p_winner': 'p_elo'})

elo['k_w'] = elo['winner_name'].map(key_from_full)
elo['k_l'] = elo['loser_name'].map(key_from_full)
elo['pair'] = elo.apply(lambda r: '|'.join(sorted([r.k_w, r.k_l])), axis=1)

odds = load_odds(years=(2024, 2025)).dropna(subset=['Winner', 'Loser', 'Date'])
odds['k_w'] = odds['Winner'].map(key_from_short)
odds['k_l'] = odds['Loser'].map(key_from_short)
odds['date'] = pd.to_datetime(odds['Date'], errors='coerce')
odds = odds.dropna(subset=['date'])
odds['pair'] = odds.apply(lambda r: '|'.join(sorted([r.k_w, r.k_l])), axis=1)

j = elo.merge(odds[['pair', 'tour', 'date', 'k_w', 'k_l', 'PSW', 'PSL']],
              on=['pair', 'tour'], how='left', suffixes=('', '_o'))
j['gap'] = (j['date_o'] - j['date']).dt.days
j = j[(j['gap'] >= -3) & (j['gap'] <= 21)]
j = j.sort_values('gap').drop_duplicates(subset=['pair', 'tour', 'date'], keep='first')
j = j[(j['k_w_o'] == j['k_w']) & j[['PSW', 'PSL']].notna().all(axis=1)]
j['p_mkt'] = devig_power(j['PSW'].values.astype(float), j['PSL'].values.astype(float))
j = j.dropna(subset=['p_mkt'])

print(f"n rows joined: {len(j):,}")
for col, name in [('p_elo', 'Elo'), ('p_mkt', 'Market')]:
    pr = np.clip(j[col].values, 1e-9, 1 - 1e-9)
    acc = float((pr > 0.5).mean())
    logloss = float(-np.log(pr).mean())
    brier = float(((1 - pr) ** 2).mean())
    print(f'{name:15s} acc={acc:.4f}  logloss={logloss:.4f}  brier={brier:.4f}')

# tour split
for tour in ('atp', 'wta'):
    sub = j[j['tour'] == tour]
    pr = np.clip(sub['p_elo'].values, 1e-9, 1 - 1e-9)
    print(f"  {tour} n={len(sub):,} logloss={-np.log(pr).mean():.4f}")
