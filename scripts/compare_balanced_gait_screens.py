"""Measured final-arm comparison; no acceptance from training rewards.

Read complete21-case screens and all350 settled walking samples in each of
three common-physics straight commands. Keep failed cases. Report raw5N duty
and separately two-sample-debounced complete cycles; neither is a gait label.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

LEGS = ("FL", "FR", "RL", "RR")
CASES = ("normal", "retained08", "target10")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cycles(flags, times):
    """Require2consecutive samples to confirm a state; time is confirmation time."""
    state = pending = None
    repeat = 0
    touchdown = liftoff = None
    result = []
    for flag, time in zip(flags, times):
        if flag == pending:
            repeat += 1
        else:
            pending, repeat = flag, 1
        if repeat < 2 or flag == state:
            continue
        previous, state = state, flag
        if previous is None:
            continue  # Initial partial stance/flight does not create a cycle.
        if flag:
            if touchdown is not None and liftoff is not None:
                period = time - touchdown
                result.append({"period_s": period, "stance_s": liftoff - touchdown,
                               "swing_s": time - liftoff})
            touchdown, liftoff = time, None
        elif touchdown is not None:
            liftoff = time
    return result


def measure(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        complete = list(csv.DictReader(stream))
    require(len(complete) == 900, "Require the complete unchanged900-step CSV")
    rows = [row for row in complete if row["phase"] == "walk" and float(row["time_s"]) > 5.0]
    require(len(rows) == 350, "Require all350 samples in the predeclared settled walking window")
    times = [float(row["time_s"]) for row in rows]
    require(all(abs(t - (5.02 + .02*i)) < 1e-9 for i,t in enumerate(times)), "Missing/reordered samples")
    contacts, duties, foot_cycles = [], {}, {}
    for leg in LEGS:
        force = [float(row[f"{leg}_foot_fz"]) for row in rows]
        require(all(math.isfinite(value) for value in force), "Nonfinite current foot forces")
        flags = [value > 5.0 for value in force]
        require(all(row[f"{leg}_foot_vertical_contact"] == str(int(flag)) for row,flag in zip(rows,flags)),
                "Recorded current vertical-contact flags differ from actual force")
        contacts.append(flags)
        duties[leg] = sum(flags)/350
        records = cycles(flags,times)
        foot_cycles[leg] = {"complete_cycles": len(records), "cycles": records,
                           "period_mean_s": statistics.mean(r["period_s"] for r in records) if records else None,
                           "short_period_below_150ms_count": sum(r["period_s"] < .15 for r in records)}
    return {"csv": str(Path(path).resolve()), "csv_sha256": sha(path), "samples":350,
            "current_vertical_force_threshold_n_exclusive":5., "duty":duties,
            "diagonal_duty_gap":abs((duties["FL"]+duties["RR"]-duties["FR"]-duties["RL"])/2),
            "exact_diagonal_support_fraction":sum(tuple(c[i] for c in contacts) in ((True,False,False,True),(False,True,True,False)) for i in range(350))/350,
            "sampled_all_feet_flight_fraction":sum(not any(c[i] for c in contacts) for i in range(350))/350,
            "debounced_cycles":foot_cycles}


def load_screen(path, arm):
    summary=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(summary["protocol"] == "balanced_gait_candidate21_screen_v1" and
            summary["status"] == "completed_screening_only" and summary["arm"] == arm and len(summary["rows"]) == 21,
            "Require completed actual21-case matching arm")
    expected={("recovery",case,20260909) for case in (*CASES,"left","right","push05")}
    expected |= {("original",case,seed) for case in ("normal","retained08","left","right","push05")
                 for seed in (20260909,20260910,20260911)}
    require({(r["physics"],r["case"],r["seed"]) for r in summary["rows"]} == expected,
            "Missing/duplicated prescribed screen identity")
    measured={}
    for row in summary["rows"]:
        require(sha(row["report"]) == row["report_sha256"], "Report changed")
        report=json.loads(Path(row["report"]).read_text(encoding="utf-8-sig"))
        require(report["checkpoint_sha256"] == summary["checkpoint_sha256"] and
                report["retention_passed"] is row["passed"], "Report/summary identity mismatch")
        if row["physics"] == "recovery" and row["case"] in CASES:
            require(row["seed"] == 20260909 and row["case"] not in measured, "Unexpected/duplicate comparison case")
            csv_path=report["artifacts"]["csv"]
            require(sha(csv_path) == row["csv_sha256"], "CSV changed")
            measured[row["case"]]=measure(csv_path)
    require(set(measured) == set(CASES), "Missing declared straight command")
    return summary,measured


def compare(control_path, balanced_path):
    a,am=load_screen(control_path,"control")
    b,bm=load_screen(balanced_path,"balanced")
    mean_a=statistics.mean(am[c]["diagonal_duty_gap"] for c in CASES)
    mean_b=statistics.mean(bm[c]["diagonal_duty_gap"] for c in CASES)
    checks={"balanced_all21_old_physical_criteria":all(row["passed"] for row in b["rows"]),
            "mean_duty_gap_reduction_at_least25percent":mean_b <= .75*mean_a,
            "no_case_gap_worsens_over_point05":all(bm[c]["diagonal_duty_gap"]-am[c]["diagonal_duty_gap"] <= .05 for c in CASES),
            "all_feet_at_least_two_complete_debounced_cycles":all(bm[c]["debounced_cycles"][leg]["complete_cycles"] >= 2 for c in CASES for leg in LEGS),
            "no_more_short_cycles_per_case_foot":all(bm[c]["debounced_cycles"][leg]["short_period_below_150ms_count"] <= am[c]["debounced_cycles"][leg]["short_period_below_150ms_count"] for c in CASES for leg in LEGS)}
    return {"protocol":"matched_gait21_comparison_v1","control_summary":str(control_path),"balanced_summary":str(balanced_path),
            "control_summary_sha256":sha(control_path),"balanced_summary_sha256":sha(balanced_path),
            "source_sha256":sha(__file__),"control":am,"balanced":bm,"mean_duty_gap_control":mean_a,
            "mean_duty_gap_balanced":mean_b,"checks":checks,"diagnostic_selection_passed":all(checks.values()),
            "promotion_performed":False,"scope":"Repeated development screening only. Two-sample40ms persistence confirms contact transitions; initial/incomplete cycles excluded.150ms is a diagnostic short-period flag, not a gait definition. No visual/fresh-start/full-flow/fast-running acceptance."}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control",type=Path)
    parser.add_argument("balanced",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    require(args.output.resolve().drive.upper()=="E:" and not args.output.exists(),"Use a fresh E-drive output")
    result=compare(args.control,args.balanced)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({"checks":result["checks"],"output":str(args.output)},indent=2))
