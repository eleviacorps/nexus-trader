import torch, os
from pathlib import Path

for p in ['models/v24/diffusion_unet1d_v2_6m_phase06.pt', 'models/v24/diffusion_unet1d_v2_6m.pt']:
    if os.path.exists(p):
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        print(f'{p}: epoch={ckpt["epoch"]}, val_loss={ckpt.get("best_val_loss", "?")}, phase={ckpt.get("phase", "?")}')