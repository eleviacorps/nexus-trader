import sys
sys.path.insert(0, ".")
from v30.models.selector.diffusion_selector import DiffusionSelector
import torch

m = DiffusionSelector(144, 20, 64, 256, 100, 0.1).cuda()
ctx = torch.randn(4, 144).cuda()
paths = torch.randn(4, 64, 20).cuda()

# Debug the denoiser directly
state_emb = m.state_encoder(ctx)
path_emb = m.path_encoder(paths)
state_expanded = state_emb.unsqueeze(1).expand(-1, 64, -1)
attention_input = torch.cat([path_emb, state_expanded], dim=-1)
attention_scores = m.path_attention(attention_input).squeeze(-1)
temperature = torch.nn.functional.softplus(m.log_temperature).clamp(min=0.01, max=2.0)
weights = torch.nn.functional.softmax(attention_scores / temperature, dim=-1)
dist_summary = m.dist_encoder(paths)
condition = torch.cat([state_emb, dist_summary], dim=-1)
print("condition shape:", condition.shape)

t = torch.randint(0, 100, (4,), device='cuda').long()
noisy_ret = torch.randn(4, 1).cuda()

time_emb = m.denoiser.time_embed(t)
print("time_emb shape:", time_emb.shape)
condition_film = m.denoiser.cond_proj(condition)
print("condition_film shape:", condition_film.shape)
cond_filmed = m.denoiser.film_time(condition, time_emb)
print("cond_filmed shape:", cond_filmed.shape)
print("noisy_ret shape:", noisy_ret.shape)