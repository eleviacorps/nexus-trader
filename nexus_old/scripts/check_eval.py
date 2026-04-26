import torch
import json
import numpy as np
from pathlib import Path

ckpt = torch.load('models/v24/diffusion_unet1d_v2_6m.pt', map_location='cpu', weights_only=False)
print(f'Checkpoint epoch: {ckpt["epoch"]}')
best_val = ckpt.get("best_val_loss", "N/A")
print(f'Best val_loss: {best_val}')