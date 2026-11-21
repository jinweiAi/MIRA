"""Portable JSONL data interface; images remain outside the code repository."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def read_manifest(path, *, training=False, limit=None, check_images=True):
    path = Path(path).resolve()
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")
    rows, seen = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "id" not in row or not str(row["id"]):
            raise ValueError(f"line {line_number}: a unique id is required")
        if str(row["id"]) in seen:
            raise ValueError(f"line {line_number}: duplicate id")
        seen.add(str(row["id"]))
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"line {line_number}: question must be nonempty")
        if training and (not isinstance(row.get("answer"), str) or not row["answer"].strip()):
            raise ValueError(f"line {line_number}: training needs a nonempty answer")
        images = row.get("images")
        if not isinstance(images, list) or not images or any(not isinstance(p, str) for p in images):
            raise ValueError(f"line {line_number}: images must be a nonempty list of paths")
        row = dict(row, images=[str((path.parent / p).resolve()) for p in images])
        if check_images:
            for image in row["images"]:
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
        options = row.get("options")
        if options is not None:
            if (not isinstance(options, list) or not options
                    or any(not isinstance(o, dict) or not isinstance(o.get("text"), str)
                           or o.get("label") not in list("ABCDEFGH") for o in options)
                    or len({o["label"] for o in options}) != len(options)):
                raise ValueError(f"line {line_number}: options need unique A-H labels and text")
        rows.append(row)
        if limit and len(rows) == limit:
            break
    if not rows:
        raise ValueError("manifest is empty")
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Prepared JSONL manifest")
    p.add_argument("--output", required=True, help="Validated JSONL with resolved image paths")
    p.add_argument("--training", action="store_true")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = read_manifest(args.input, training=args.training)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"rows": len(rows), "output": str(output)}))
if __name__ == "__main__":
    main()
