from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sync_server_to_local import sync_paths  # noqa: E402


DEFAULT_PATHS = ["outputs/evaluation", "models/tft", "outputs/logs"]


def api_url(base_url: str, remote_path: str) -> str:
    normalized_base = base_url.rstrip("/")
    normalized_path = remote_path.strip("/").replace("\\", "/")
    return f"{normalized_base}/api/contents/{normalized_path}"


def exists(session: requests.Session, base_url: str, remote_path: str) -> bool:
    response = session.get(api_url(base_url, remote_path), params={"content": 0}, timeout=60)
    return response.status_code == 200


def complete_for_tag(session: requests.Session, base_url: str, remote_root: str, run_tag: str) -> bool:
    required = [
        f"{remote_root}/outputs/evaluation/training_summary_{run_tag}.json",
        f"{remote_root}/outputs/evaluation/walkforward_report_{run_tag}.json",
        f"{remote_root}/outputs/evaluation/backtest_report_{run_tag}.json",
    ]
    return all(exists(session, base_url, path) for path in required)


def tail_log(session: requests.Session, base_url: str, remote_root: str, relative_path: str, lines: int = 6) -> str:
    remote_path = f"{remote_root}/{relative_path.strip('/')}"
    response = session.get(api_url(base_url, remote_path), params={"content": 1}, timeout=60)
    if response.status_code != 200:
        return ""
    payload = response.json()
    content = payload.get("content") or ""
    return "\n".join(content.splitlines()[-lines:])


def notify_beep() -> None:
    try:
        import winsound  # type: ignore

        winsound.MessageBeep()
        winsound.MessageBeep()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor remote Nexus cloud runs and sync artifacts when complete.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--remote-root", default="nexus")
    parser.add_argument("--local-root", default=str(ROOT))
    parser.add_argument("--run-tags", nargs="+", required=True, help="Example: mh12_full_v2 mh12_recent_v2")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--sync-paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--beep", action="store_true")
    args = parser.parse_args()

    session = requests.Session()
    session.params = {"token": args.token}
    completed: set[str] = set()

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] polling remote runs...", flush=True)
        for run_tag in args.run_tags:
            if run_tag in completed:
                continue
            try:
                done = complete_for_tag(session, args.base_url, args.remote_root, run_tag)
            except requests.RequestException as exc:
                print(f"  {run_tag}: network error: {exc}", flush=True)
                continue
            if done:
                print(f"  {run_tag}: complete", flush=True)
                completed.add(run_tag)
                continue
            log_tail = tail_log(session, args.base_url, args.remote_root, f"outputs/logs/{run_tag}_train.log")
            print(f"  {run_tag}: running", flush=True)
            if log_tail.strip():
                print(log_tail, flush=True)

        if len(completed) == len(args.run_tags):
            print("all requested run tags completed; syncing artifacts...", flush=True)
            sync_paths(
                base_url=args.base_url,
                token=args.token,
                local_root=Path(args.local_root).resolve(),
                remote_root=args.remote_root,
                paths=args.sync_paths,
            )
            if args.beep:
                notify_beep()
            summary_path = Path(args.local_root).resolve() / "outputs" / "logs" / "monitor_cloud_run_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "completed_run_tags": sorted(completed),
                        "synced_paths": args.sync_paths,
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"summary written to {summary_path}", flush=True)
            return 0

        time.sleep(max(10, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
