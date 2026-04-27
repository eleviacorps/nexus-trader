import sys
sys.path.insert(0, 'C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged')
import torch
from MMFPS.generator.constrained_diffusion_generator import ConstrainedDiffusionGenerator, DiffusionGeneratorConfig

try:
    ckpt = torch.load('C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints/checkpoint_step_5000.pt', map_location='cuda', weights_only=False)
    print(f'Step: {ckpt.get("step", "unknown")}')
    model = ConstrainedDiffusionGenerator(DiffusionGeneratorConfig())
    model.load_state_dict(ckpt['model'])
    print('Loaded OK')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

try:
    ckpt2 = torch.load('C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints/checkpoint_step_4000.pt', map_location='cuda', weights_only=False)
    print(f'Step 4k: {ckpt2.get("step", "unknown")}')
except Exception as e:
    print(f'Step 4k error: {e}')

try:
    ckpt3 = torch.load('C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints/checkpoint_step_3000.pt', map_location='cuda', weights_only=False)
    print(f'Step 3k: {ckpt3.get("step", "unknown")}')
except Exception as e:
    print(f'Step 3k error: {e}')