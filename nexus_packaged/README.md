# Nexus Trader Packaged Runtime (`nexus_packaged`)

This folder contains a fully isolated production runtime for Nexus Trader v27.1.
It does not import from `src.*` at runtime.

## Quick Start

1. Install dependencies:
```bash
pip install -r nexus_packaged/requirements.txt
```

2. Set model key:
```bash
export NEXUS_MODEL_KEY="your-secret-passphrase"
```

3. Build/validate data:
```bash
python -m nexus_packaged.core.feature_pipeline
```

4. Launch:
```bash
python -m nexus_packaged.main --no-mt5 --no-webview --paper
```

Optional local key file (already gitignored):
```bash
echo 'NEXUS_MODEL_KEY=your-secret-passphrase' > nexus_packaged/.env.local
```

## Windows Build (.exe)

PowerShell-native build (no bash required):

```powershell
powershell -ExecutionPolicy Bypass -File nexus_packaged/build_exe.ps1
```

Dry run without compilation:

```powershell
powershell -ExecutionPolicy Bypass -File nexus_packaged/build_exe.ps1 -SkipCompile
```

## CLI

```bash
python -m nexus_packaged.main                         # normal launch
python -m nexus_packaged.main --no-mt5                # skip MT5
python -m nexus_packaged.main --no-webview            # force ASCII chart
python -m nexus_packaged.main --backtest-only         # run headless backtest
python -m nexus_packaged.main --rebuild-data          # rebuild feature data and exit
python -m nexus_packaged.main --encrypt-model PATH    # encrypt model to protection/model_enc.bin
python -m nexus_packaged.main --paper                 # force paper mode
```

## API (External LLM / Agent Interface)

The local API server is the official integration layer for external agents.
Default URL: `http://127.0.0.1:8765`.

### Endpoints

- `GET /health`
- `GET /state`
- `GET /prediction`
- `GET /news`
- `GET /trades/open`
- `GET /trades/history?limit=50&source=auto`
- `POST /auto_trade/toggle`
- `POST /auto_trade/config`
- `POST /trade/manual`
- `POST /trade/close`
- `POST /backtest/run`
- `GET /backtest/results/{job_id}`
- `GET /mt5/account`
- `POST /mt5/account/config`
- `POST /mt5/account/connect`
- `POST /mt5/account/disconnect`
- `GET /hybrid` (TradingView Lightweight Charts dashboard)
- `GET /paths`
- `GET /signal`
- `GET /ohlc`

### cURL examples

```bash
curl http://127.0.0.1:8765/health
```

```bash
curl http://127.0.0.1:8765/state
```

```bash
curl -X POST http://127.0.0.1:8765/auto_trade/toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

```bash
curl -X POST http://127.0.0.1:8765/trade/manual \
  -H "Content-Type: application/json" \
  -d '{
    "direction":"BUY",
    "lot_size":0.01,
    "entry_type":"market",
    "entry_price":null,
    "sl":2300.0,
    "tp":2320.0,
    "comment":"manual_api"
  }'
```

```bash
curl -X POST http://127.0.0.1:8765/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "start_date":"2024-01-01",
    "end_date":"2024-12-31",
    "timeframe_minutes":15
  }'
```

```bash
curl -X POST http://127.0.0.1:8765/mt5/account/config \
  -H "Content-Type: application/json" \
  -d '{
    "login":106193715,
    "password":"your_password",
    "server":"MetaQuotes-Demo",
    "execution_enabled":false,
    "reconnect_now":true,
    "persist_to_settings":false
  }'
```

### Python client (`httpx`)

```python
import httpx

BASE = "http://127.0.0.1:8765"
with httpx.Client(timeout=10.0) as client:
    health = client.get(f"{BASE}/health").json()
    state = client.get(f"{BASE}/state").json()
    toggle = client.post(f"{BASE}/auto_trade/toggle", json={"enabled": True}).json()
    print(health, state["signal"], toggle)
```

## Auto Trade via API

Enable:
```bash
POST /auto_trade/toggle {"enabled": true}
```

Configure:
```bash
POST /auto_trade/config { ... full AutoTradeConfig ... }
```

## Manual Trade via API

```bash
POST /trade/manual
```
with `ManualOrderRequest` payload.

## Extensibility

### Add a new model loader

Create a class implementing:
- `nexus_packaged.core.diffusion_loader.BaseModelLoader.load()`
- `predict(context_window)`
- `warm_up()`
- `is_loaded`

Then wire it in `nexus_packaged/main.py`.

### Add a new asset

1. Update `nexus_packaged/config/settings.json`:
   - `data.symbol`
2. Rebuild data:
```bash
python -m nexus_packaged.main --rebuild-data
```

### Add a new news source

Append source object to:
- `settings.json -> news.rss_sources`

## MT5 Account Switching

You can switch MT5 accounts at runtime without restarting:

1. In TUI press `s` to open **MT5 Account Settings**.
2. Enter `login`, `password`, `server`.
3. Choose:
- `Apply Runtime + Reconnect` for temporary switch (session only).
- `Apply + Persist + Reconnect` to also save under `settings.json -> mt5`.

Equivalent API path:
- `POST /mt5/account/config`

## Hybrid Chart + EV Decision

- Decision logic is EV-based (not majority-vote):
  - `ev = mean((final_price - entry_price) / entry_price)`
  - dispersion-aware confidence from `ev / (std + 1e-6)` normalized to `[0, 1]`
- Open browser dashboard from TUI via key `w`, or directly:
```bash
http://127.0.0.1:8765/hybrid
```
- TUI status line now includes: `SIGNAL | CONF | EV | STD`.

## Security

- Model weights are encrypted at rest (AES-256-GCM).
- Decryption is in-memory only.
- Key is derived from `NEXUS_MODEL_KEY` via PBKDF2-HMAC-SHA256.
- Optional executable integrity check blocks inference on tamper detection.
