"""Add per-file row counts (CSV data rows, excluding header) to manifest.json."""
import json
import csv
import os

MANIFEST = r"C:\dev\dropshotverse\dropshotdata\manifest\manifest.json"
GH_ROOT = r"C:\dev\dropshotverse\dropshotdata\raw\github"

d = json.load(open(MANIFEST, encoding="utf-8"))

for entry in d["files"]:
    path = entry["path"]
    if not path.lower().endswith(".csv"):
        entry["row_count"] = None
        continue
    fp = os.path.join(GH_ROOT, path)
    if not os.path.exists(fp):
        fp = os.path.join(r"C:\dev\dropshotverse\dropshotdata\raw\huggingface", path)
    if not os.path.exists(fp):
        entry["row_count"] = None
        continue
    with open(fp, encoding="utf-8", errors="replace", newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1  # minus header
    entry["row_count"] = max(n, 0)

total_rows = sum(e["row_count"] for e in d["files"] if e.get("row_count") is not None)
d["total_csv_data_rows_all_files"] = total_rows

with open(MANIFEST, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2)

print("Done. Total CSV data rows across all 465 CSVs:", total_rows)

# Also emit a flat CSV manifest for human readability
import csv as csvmod
csv_out = r"C:\dev\dropshotverse\dropshotdata\manifest\manifest.csv"
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    w = csvmod.writer(f)
    w.writerow(["path", "priority", "gh_size_bytes", "hf_size_bytes", "gh_sha256", "hf_sha256",
                "mirrors_agree", "row_count", "fetched_at_utc"])
    for e in d["files"]:
        w.writerow([e["path"], e["priority"], e.get("gh_size_bytes"), e.get("hf_size_bytes"),
                    e.get("gh_sha256"), e.get("hf_sha256"), e.get("mirrors_agree"),
                    e.get("row_count"), e.get("fetched_at_utc")])
print("Wrote", csv_out)
