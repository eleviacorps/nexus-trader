from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.project_config import FINAL_DASHBOARD_HTML_PATH, FUTURE_BRANCHES_PATH, LATEST_MARKET_SNAPSHOT_PATH, MODEL_MANIFEST_PATH
from src.service.app import ModelServer

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except ImportError:  # pragma: no cover
    FastMCP = None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


if FastMCP is not None:
    mcp = FastMCP('Nexus Trader')

    @mcp.tool()
    def get_model_metadata() -> dict[str, Any]:
        return _read_json(MODEL_MANIFEST_PATH)

    @mcp.tool()
    def get_latest_market_snapshot() -> dict[str, Any]:
        return _read_json(LATEST_MARKET_SNAPSHOT_PATH)

    @mcp.tool()
    def get_latest_branches() -> Any:
        return _read_json(FUTURE_BRANCHES_PATH)

    @mcp.tool()
    def predict_direction(sequence: list[list[float]]) -> dict[str, Any]:
        server = ModelServer()
        return server.predict(sequence).model_dump()

    @mcp.resource('nexus://dashboard')
    def dashboard_html() -> str:
        if not FINAL_DASHBOARD_HTML_PATH.exists():
            return '<html><body><h1>Nexus Trader dashboard not generated yet.</h1></body></html>'
        return FINAL_DASHBOARD_HTML_PATH.read_text(encoding='utf-8')


def main() -> int:
    if FastMCP is None:
        raise SystemExit('The `mcp` package is required to run the Nexus Trader MCP server. Install it with `pip install mcp`.')
    mcp.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
