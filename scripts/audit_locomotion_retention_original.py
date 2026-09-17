"""Authenticate the one original-physics instrumentation control without simulation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "evaluations/20260917-speed-retention-control3947/normal-seed20260909/model_3947_stand_walk_stop.json"
NEW = ROOT / "evaluations/20260917-locomotion-recovery-retention-v1/original-normal-seed20260909/model_3947_stand_walk_stop.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old, new = [json.loads(p.read_text(encoding="utf-8-sig")) for p in (OLD, NEW)]
    summary = json.loads((ROOT / "evaluations/20260917-speed-retention-control3947/summary.json").read_text(encoding="utf-8-sig"))
    rows = [r for r in summary["rows"] if Path(r["report"]).resolve() == OLD.resolve()]
    if len(rows) != 1 or rows[0]["report_sha256"].lower() != sha(OLD):
        raise ValueError("Historical control identity mismatch")
    if new["retention_experiment"]["physics_mode"] != "original" or new["baseline_metric_protocol_version"] != old["protocol_version"]:
        raise ValueError("Wrong protocol/mode")
    checked = [k for k in old if k not in ("protocol_version", "artifacts")]
    for key in checked:
        if old[key] != new[key]:
            raise ValueError("Original physical result changed: " + key)
    old_csv, new_csv = [Path(r["artifacts"]["csv"]) for r in (old, new)]
    if sha(old_csv) != sha(new_csv) or old["artifacts"]["video"] is not None or new["artifacts"]["video"] is not None:
        raise ValueError("Full900-row CSV equality/video mode failed")
    result = {"protocol": "retention_instrumentation_original_exact_audit_v1", "exact": True,
              "old_report": str(OLD), "old_sha256": sha(OLD), "new_report": str(NEW), "new_sha256": sha(NEW),
              "common_report_keys_exact": checked, "full900row_csv_sha256": sha(new_csv),
              "scope": "Only original0.5 seed20260909 instrumentation equivalence; not recovery-physics retention or full flow"}
    with (NEW.parent / "original_exact_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
