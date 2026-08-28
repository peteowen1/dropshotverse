import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.loads(Path(r"C:\dev\dropshotverse\dropshotdata\ticket09\traces.json").read_text())

fig, axes = plt.subplots(len(data), 1, figsize=(9, 3.2 * len(data)), sharex=False)
if len(data) == 1:
    axes = [axes]

for ax, r in zip(axes, data):
    tr = r["trace"]
    x = [t["game_idx"] for t in tr]
    a = [t["p_a"] for t in tr]
    b = [t["p_b"] for t in tr]
    ax.plot(x, a, label="A: frozen skill", color="#2166ac", linewidth=2)
    ax.plot(x, b, label="B: live-updating skill", color="#b2182b", linewidth=2, linestyle="--")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    # mark set boundaries
    prev_set = 1
    for t in tr:
        if t["set"] != prev_set:
            ax.axvline(t["game_idx"], color="black", linewidth=0.6, alpha=0.4)
            prev_set = t["set"]
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{r['match_id']}: {r['label']}", fontsize=10)
    ax.set_ylabel("P(player 1 wins match)")
    ax.legend(fontsize=8, loc="lower right")

axes[-1].set_xlabel("games completed")
fig.suptitle("Ticket 09 — frozen (A) vs live-updating (B) win probability\n"
             "real point-by-point data, 2026 US Open Fan Week", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = r"C:\dev\dropshotverse\dropshotdata\ticket09\traces.png"
fig.savefig(out, dpi=130)
print("wrote", out)
