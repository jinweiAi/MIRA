"""Run the external VLMEvalKit with MIRA registered at runtime."""
import argparse
from functools import partial
import runpy
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlmeval-root", required=True, help="Path to a VLMEvalKit checkout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=16)
    args, extra = p.parse_known_args()
    root = Path(args.vlmeval_root).resolve()
    if not (root / "run.py").is_file():
        raise FileNotFoundError(root / "run.py")
    sys.path.insert(0, str(root))
    from vlmeval.config import supported_VLM
    from mira.vlmeval import MIRAModel
    supported_VLM["MIRA"] = partial(MIRAModel, checkpoint=str(Path(args.checkpoint).resolve()),
        model=args.model, device=args.device, max_new_tokens=args.max_new_tokens)
    sys.argv = [str(root / "run.py"), "--model", "MIRA", *extra]
    runpy.run_path(str(root / "run.py"), run_name="__main__")

if __name__ == "__main__":
    main()
