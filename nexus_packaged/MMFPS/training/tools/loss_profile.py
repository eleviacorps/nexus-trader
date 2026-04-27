import sys
sys.path.insert(0, 'C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged')
import torch
from torch.cuda.amp import autocast, GradScaler
from MMFPS.generator.constrained_diffusion_generator import ConstrainedDiffusionGenerator, DiffusionGeneratorConfig

model = ConstrainedDiffusionGenerator(DiffusionGeneratorConfig()).to('cuda').train()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()
ctx = torch.randn(64, 144, device='cuda')
reg = torch.zeros(64, 64, device='cuda')
quant = torch.zeros(64, 64, device='cuda')
targets = torch.randn(64, 1, 20, 144, device='cuda')

for i in range(30):
    opt.zero_grad()
    with autocast():
        out = model(ctx, reg, quant, targets)
        loss = out.diversity_loss
    loss_val = loss.item()
    if loss_val > 100 or loss_val < -100:
        print(f'Step {i}: EXPLOSION loss={loss_val:.4f}')
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(opt)
    scaler.update()
    if i % 10 == 0:
        print(f'Step {i}: loss={loss_val:.4f}')

print('Done - stable' if all(True for _ in range(1)) else 'Had explosions')