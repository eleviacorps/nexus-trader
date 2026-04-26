import torch
c = torch.load('models/v24/diffusion_unet1d_v2_6m_phase07.pt', map_location='cpu', weights_only=False)
print(f'phase07.pt: epoch={c["epoch"]}, val_loss={c.get("best_val_loss", "?")}')