"""Strict and normalized multiple-choice micro accuracy; no judge-model calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from .answer_parser import VERSION, parse
from .data import read_manifest

def score(rows, records):
    expected = {str(r["id"]) for r in rows}
    predictions = {}
    for record in records:
        key = str(record["id"])
        if key in predictions:
            raise ValueError(f"Duplicate prediction id: {key}")
        predictions[key] = record
    if set(predictions) != expected:
        raise ValueError(f"Prediction coverage mismatch: missing={len(expected-set(predictions))}, "
                         f"extra={len(set(predictions)-expected)}")
    audit, groups = [], defaultdict(list)
    for row in rows:
        if not row.get("options"):
            raise ValueError("MCQ scoring requires options; use the official evaluator for other tasks")
        valid = {o["label"] for o in row["options"]}
        if row.get("answer") not in valid:
            raise ValueError(f"Gold label is not a legal option: {row['id']}")
        text = predictions[str(row["id"])]["prediction"]
        strict = re.fullmatch(r"\s*[\(\[]?([A-H])[\)\]]?[.!?,;]?\s*", str(text))
        strict = strict.group(1) if strict and strict.group(1) in valid else None
        # The parser accepts prediction and options only; never gold.
        normalized, rule = parse(text, row["options"])
        item = {"id": row["id"], "strict": strict, "normalized": normalized, "rule": rule,
                "strict_correct": strict == row["answer"],
                "normalized_correct": normalized == row["answer"],
                "unparsed": normalized is None}
        audit.append(item)
        groups[str(row.get("category", "uncategorized"))].append(item)
    def aggregate(items):
        n = len(items)
        return {"n": n, "strict_letter_accuracy": 100 * sum(x["strict_correct"] for x in items) / n,
                "normalized_accuracy": 100 * sum(x["normalized_correct"] for x in items) / n,
                "unparsed": sum(x["unparsed"] for x in items)}
    return {"micro": aggregate(audit), "categories": {k: aggregate(v) for k, v in groups.items()},
            "parser_version": VERSION, "unparsed_remain_in_denominator": True}, audit

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = read_manifest(args.manifest, check_images=False)
    records = [json.loads(s) for s in Path(args.predictions).read_text().splitlines() if s.strip()]
    result, audit = score(rows, records)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".audit.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in audit))
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
