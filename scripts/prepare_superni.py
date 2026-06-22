from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cpmoe.data import SUPERNI_TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CP-MoE SuperNI task JSON files.")
    parser.add_argument("--source", required=True, help="Folder containing SuperNI/NaturalInstructions task JSONs")
    parser.add_argument("--output", default="data/raw/superni")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    missing = []
    copied = []
    for task_name in SUPERNI_TASKS:
        match = _find_task_file(source, task_name)
        if match is None:
            missing.append(task_name)
            continue
        target = output / f"{task_name}.json"
        shutil.copyfile(match, target)
        copied.append((task_name, str(match)))
    print(f"Copied {len(copied)} task files to {output}")
    for task_name, path in copied:
        print(f"  {task_name}: {path}")
    if missing:
        print("Missing tasks:")
        for task_name in missing:
            print(f"  {task_name}")
        raise SystemExit(1)


def _find_task_file(root: Path, task_name: str) -> Path | None:
    candidates = [
        root / f"{task_name}.json",
        root / "tasks" / f"{task_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(f"{task_name}.json"))
    return matches[0] if matches else None


if __name__ == "__main__":
    main()