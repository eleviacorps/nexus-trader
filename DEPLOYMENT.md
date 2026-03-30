**Deployment**

This repo now has a minimal production inference surface for the fused Nexus Trader model.

**What It Expects**

- A trained checkpoint at `models/tft/final_tft.ckpt` or the legacy checkpoint path.
- A model manifest at `models/tft/model_manifest.json`.
- Python dependencies from `requirements-prod.txt`.

**Train And Export The Bundle**

```bash
python scripts/build_fused_artifacts.py
python scripts/train_fused_tft.py --epochs 5 --batch-size 256
```

That training step writes:

- `models/tft/final_tft.ckpt`
- `models/tft/model_manifest.json`
- `outputs/evaluation/training_summary.json`
- `outputs/evaluation/tft_metrics.json`
- `outputs/evaluation/calibration_report.json`
- `outputs/evaluation/feature_importance.json`

**Run The API Locally**

```bash
pip install -r requirements-prod.txt
python -m src.service.app
```

Default endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

`/predict` expects a JSON body shaped like:

```json
{
  "sequence": [[0.0, 0.0, "... 100 features ..."], "... 120 rows total ..."]
}
```

**Container Build**

```bash
docker build -t nexus-trader-api .
docker run --rm -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  nexus-trader-api
```

**Production Notes**

- The current API is suitable for serving a trained checkpoint, not for online training.
- Training should stay on the ROCm host until a dedicated ROCm container base image is chosen.
- Accuracy targets in live markets should be validated with walk-forward backtests and leakage checks before any trading use.
