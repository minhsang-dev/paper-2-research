"""Smoke demo (synthetic only): attack family 5 is present on days 0-4 (about 0.4 % of rows), absent on days 5-9 and
returns on day 10. Counts the rows of family 5 in each policy's context after every update and reads the day-10
recall of family 5 from the replay records. Writes absence_demo_results.json. Not a result about real data; ThreeBin
runs with the placeholder defaults of DESIGN_CHOICES.md (nothing tuned).
Run: python run_absence_demo.py"""
import json
from pathlib import Path

import threebin  # noqa: F401  (adds protocol_code/replay to sys.path)
from models import LightGBMModel  # noqa: E402
from policies import FIFO, CBRSFamily, DualFIFO  # noqa: E402
from replay import run_replay  # noqa: E402
from synth import SynthStream, absence_probs, recording
from threebin import ThreeBin

K, B, ETA, TAU, A_FP, N_DAYS, PER_DAY, N_EVAL = 1000, 600, 0.2, 0, 50, 11, 4000, 1000
SEEDS = (0, 1, 2)
RARE, ABSENT, RETURN = 5, list(range(5, 10)), 10
LGBM = dict(n_estimators=50, verbosity=-1)
POLICIES = {"FIFO": (FIFO, {}), "DualFIFO": (DualFIFO, {}), "CBRS-family": (CBRSFamily, {}),
            "ThreeBin": (ThreeBin, {})}


def main():
    out = {"setup": dict(k=K, b=B, eta=ETA, tau=TAU, a_fp=A_FP, n_days=N_DAYS, per_day=PER_DAY, n_eval=N_EVAL,
                         seeds=SEEDS, rare_family=RARE, absent_days=ABSENT, return_day=RETURN, lgbm=LGBM,
                         threebin_defaults="ThreeBin() constructor defaults (DESIGN_CHOICES.md)"),
           "runs": []}
    for seed in SEEDS:
        stream = SynthStream(N_DAYS, absence_probs(), per_day=PER_DAY, seed=seed)
        for name, (cls, kw) in POLICIES.items():
            p = recording(cls)(k=K, seed=seed, **kw)
            recs = run_replay(stream, p, LightGBMModel(params=LGBM, seed=seed), b=B, eta=ETA, tau=TAU, a_fp=A_FP,
                              n_days=N_DAYS, n_eval=N_EVAL, seed=seed)
            ret = next(r for r in recs if r["day"] == stream.names[RETURN])
            out["runs"].append({
                "seed": seed, "policy": name,
                "rare_rows_in_context_by_day": [t["fam"].get(RARE, 0) for t in p.trace],
                "context_size_by_day": [t["n"] for t in p.trace],
                "rare_rows_after_absence": p.trace[RETURN - 1]["fam"].get(RARE, 0),
                "return_day_recall_rare": ret["recall_by_family"].get(RARE),
                "return_day_auprc_w": ret["auprc_w"]})
            print(seed, name, out["runs"][-1]["rare_rows_in_context_by_day"], out["runs"][-1]["return_day_recall_rare"])
    Path(__file__).with_name("absence_demo_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
