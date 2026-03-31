from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    commands = [
        ["python", "scripts/build_macro_context.py"],
        ["python", "scripts/build_news_embeddings.py"],
        ["python", "scripts/build_crowd_embeddings.py"],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(command, text=True, capture_output=True)
        result = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        results.append(result)
        print(json.dumps(result, indent=2))
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
