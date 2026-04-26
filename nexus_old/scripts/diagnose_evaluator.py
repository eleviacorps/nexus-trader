import torch
import numpy as np
import yaml

from nexus_packaged.v30.training.train_evaluator import PathEvaluator
from nexus_packaged.v30.training.config import V30Config
from nexus_packaged.v30.models.evaluator.structure_similarity import compute_structure_reward

config = V30Config()
model = PathEvaluator(config)
model.load_state_dict(torch.load('nexus_packaged/v30/models/evaluator/final_evaluator.pt', map_location='cpu'))
model.eval()

# Load paths
paths = np.load('nexus_packaged/v30/data/processed/v30_paths.npy')
print(f'Paths shape: {paths.shape}')

# Load features
features = np.load('data/features/diffusion_fused_6m.npy')
print(f'Features shape: {features.shape}')

# Load config
with open('nexus_packaged/v30/configs/v30_config.yaml') as f:
    cfg = yaml.safe_load(f)
lookback = cfg['data']['lookback']
horizon = cfg['data']['horizon']
feature_dim = cfg['model']['feature_dim']
print(f'Config: lookback={lookback}, horizon={horizon}, feature_dim={feature_dim}')

# Check reward distribution
batch_size = 32
idx = 5000
batch_paths = torch.FloatTensor(paths[idx:idx+batch_size]).unsqueeze(1)
batch_features = torch.FloatTensor(features[idx:idx+batch_size, -lookback:, :])

with torch.no_grad():
    pred_rewards = model(batch_paths, batch_features)
    pred_probs = torch.softmax(pred_rewards, dim=-1)
    
print(f'\nPredicted probabilities (first 5): {pred_probs[:5, 1].numpy()}')
print(f'Predicted probs mean: {pred_probs[:, 1].mean():.4f}, std: {pred_probs[:, 1].std():.4f}')

# Compute actual rewards
actual_rewards = []
for i in range(batch_size):
    p = paths[idx+i]
    f = features[idx+i, -lookback:, :]
    r = compute_structure_reward(p, f)
    actual_rewards.append(r)
actual_rewards = np.array(actual_rewards)

print(f'\nActual reward mean: {actual_rewards.mean():.4f}, std: {actual_rewards.std():.4f}')
print(f'Actual reward min: {actual_rewards.min():.4f}, max: {actual_rewards.max():.4f}')
print(f'Actual reward percentiles: 25%={np.percentile(actual_rewards, 25):.4f}, 50%={np.percentile(actual_rewards, 50):.4f}, 75%={np.percentile(actual_rewards, 75):.4f}')

# Correlation
pred_best = pred_probs[:, 1].numpy()
if actual_rewards.std() > 0:
    corr = np.corrcoef(pred_best, actual_rewards)[0, 1]
    print(f'\nPrediction-reward correlation: {corr:.4f}')

# Check if reward is binary (no gradient signal)
unique_rewards = np.unique(actual_rewards)
print(f'\nUnique reward values: {len(unique_rewards)}')
print(f'First 10 unique: {unique_rewards[:10]}')