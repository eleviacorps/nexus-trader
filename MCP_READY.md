**MCP Ready**

This repo now includes an MCP-facing server module at [server.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/mcp/server.py).

It exposes these surfaces when the `mcp` Python package is installed:
- `get_model_metadata`
- `get_latest_market_snapshot`
- `get_latest_branches`
- `predict_direction`
- resource: `nexus://dashboard`

**Run**

```bash
pip install -r requirements-prod.txt
python -m src.mcp.server
```

**Notes**

- The MCP server wraps the same local artifacts used by the FastAPI service.
- Generate the latest dashboard and branch outputs first:

```bash
python scripts/build_branching_ui.py
```

- If the dashboard has not been generated yet, the MCP dashboard resource returns a placeholder HTML page.
- This is an integration surface, not a guarantee of model quality.
