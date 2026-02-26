# A new version of the file replacing version 03
#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def load_results(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def is_success(rec: Dict[str, Any]) -> bool:
    return isinstance(rec.get("result"), dict)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def flatten_anomalies(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    res = rec.get("result") or {}
    anoms = res.get("anomalies") or []
    if not isinstance(anoms, list):
        return out

    for a in anoms:
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "sample_id": rec.get("sample_id", ""),
                "dataset": rec.get("dataset", ""),
                "frame_path": rec.get("frame_path", ""),
                "model": rec.get("model", ""),
                "anomaly_type": a.get("type") or "",
                "risk_score": safe_int(a.get("risk_score_1_to_10"), safe_int(res.get("overall_risk_score_1_to_10"), 0)),
                "risk_target": ",".join(a.get("risk_target") or []) if isinstance(a.get("risk_target"), list) else str(a.get("risk_target") or ""),
                "reasoning": (a.get("reasoning") or a.get("description") or "")[:220],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze UrbanRisk results.jsonl and print top risky anomalies.")
    parser.add_argument("--results", required=True, help="Path to results JSONL")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K anomalies by risk score")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    records = load_results(path)
    ok = [r for r in records if is_success(r)]
    err = [r for r in records if "error" in r]

    print(f"[INFO] Loaded: {len(records)} records")
    print(f"[INFO] Success: {len(ok)} | Errors: {len(err)}")

    flat: List[Dict[str, Any]] = []
    for r in ok:
        flat.extend(flatten_anomalies(r))

    if not flat:
        print("[WARN] No anomalies found in successful results.")
        return

    flat.sort(key=lambda x: x["risk_score"], reverse=True)
    top = flat[: max(1, args.top_k)]

    # Print a compact, copy-pastable table (TSV)
    headers = ["Sample ID", "Anomaly Type", "Risk Score (1-10)", "Risk Target", "Reasoning (Summary)"]
    print("\t".join(headers))
    for r in top:
        sample = r["frame_path"] or r["sample_id"]
        print("\t".join([sample, r["anomaly_type"], str(r["risk_score"]), r["risk_target"], r["reasoning"]]))

    mean_risk = sum(x["risk_score"] for x in flat) / float(len(flat))
    print(f"\n[INFO] Total anomalies: {len(flat)} | Mean anomaly risk: {mean_risk:.2f}")


if __name__ == "__main__":
    main()
