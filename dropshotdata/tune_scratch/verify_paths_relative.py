"""Every relocated constant, imported from the worktree copy, must equal the
literal the live tree still hard-codes. On this machine the worktree lives
elsewhere, so DD must resolve to the WORKTREE's dropshotdata (that is the
point), and the shape of every derived path must match the old literal's
tail. Also imports each module once to catch a NameError at import time."""
import importlib.util
import sys
from pathlib import Path

WT = Path(sys.argv[1])
SC = WT / "dropshotdata" / "scripts"
sys.path.insert(0, str(SC))
BS = chr(92)
OLD_ROOT = Path("C:" + BS + "dev" + BS + "dropshotverse" + BS + "dropshotdata")


def load(name):
    spec = importlib.util.spec_from_file_location(name, SC / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


c = load("_common")
assert c.DD == WT / "dropshotdata", c.DD
checks = {
    "simulate_tournament": [("DD", "")],
    "train_best_model": [("DD", "")],
    "bridge_tennismylife": [("DD", "")],
    "publish_model_inputs_release": [("DD", "")],
    "publish_archive_release": [("DD", "")],
    "fetch_wta_upcoming": [("DD", "")],
    "build_point_table": [("DD", "")],
    "inmatch_wp_calibration": [("DD", "")],
    "upload_blog_data": [("BLOG_DIR", "blog")],
    "check_blog_data_freshness": [("DEFAULT", "blog/live_predictions.parquet")],
    "capture_usopen": [("OUT", "raw/usopen/2026")],
    "hierarchical_model": [("ARCHIVE", "raw/github"), ("ELO_PARQUET", "elo_eval_predictions.parquet"),
                           ("PREPARED_CACHE_PATH", "tune_scratch/prepared_matches_cache.pkl")],
    "simulate_us_open_2026": [("DD", "")],
}
bad = 0
for mod, items in checks.items():
    m = load(mod)
    for attr, tail in items:
        got = Path(getattr(m, attr))
        want = (WT / "dropshotdata" / tail) if tail else (WT / "dropshotdata")
        old = (OLD_ROOT / tail) if tail else OLD_ROOT
        ok = got == want
        print(f"{'OK ' if ok else 'BAD'} {mod}.{attr} = {got}   (old literal: {old})")
        bad += not ok
        if mod == "simulate_us_open_2026":
            assert isinstance(getattr(m, attr), str), "simulate_us_open_2026.DD must stay a str"
# the two env-default literals in hierarchical_model.main() are inside the function; check the source
src = (SC / "hierarchical_model.py").read_text(encoding="utf-8")
assert 'str(_common.DD / "hier_eval_predictions.parquet")' in src and 'str(_common.DD / "hier_final_state.pkl")' in src
assert ("C:" + BS + "dev" + BS + "dropshotverse") not in src, "hierarchical_model still hard-codes the laptop path"
print("hard-coded path remaining in live modules:",
      [f.name for f in SC.glob("*.py") if ("C:" + BS + "dev" + BS + "dropshotverse") in f.read_text(encoding="utf-8") and f.stem in checks])
sys.exit(1 if bad else 0)
