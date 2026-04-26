import torch
ckpt = torch.load('models/v26/diffusion_unet1d_v2_regime.pt', map_location='cpu')
torch.save(ckpt, 'models/v26/diffusion_phase1_final.pt')
print(f'Phase 1 checkpoint preserved: epoch={ckpt["epoch"]}, val_loss={ckpt.get("best_val_loss", "?")}')