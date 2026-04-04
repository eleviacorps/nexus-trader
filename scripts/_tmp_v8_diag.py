from pathlib import Path
import json, sys
import numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.walkforward import load_model, predict_multihorizon_for_slice, apply_bucket_calibration, _combined_gate_scores, resolve_precision_gate_path, resolve_meta_gate_path
from src.training.meta_gate import load_meta_gate
from src.data.fused_dataset import DatasetSlice
from config.project_config import FUSED_FEATURE_MATRIX_PATH, TARGETS_MULTIHORIZON_PATH, GATE_CONTEXT_PATH, TFT_MODEL_DIR
manifest_path=TFT_MODEL_DIR/'model_manifest_mh12_recent_v8.json'
walk=json.loads((PROJECT_ROOT/'outputs/evaluation/walkforward_report_mh12_recent_v8.json').read_text())
model, manifest, device = load_model(manifest_path=manifest_path)
seq=int(manifest['sequence_len'])
features=np.load(FUSED_FEATURE_MATRIX_PATH, mmap_mode='r')
usable=features.shape[0]-seq
row_slice=DatasetSlice(0, usable)
targets, probs = predict_multihorizon_for_slice(model, device, row_slice, feature_path=FUSED_FEATURE_MATRIX_PATH, target_bundle_path=TARGETS_MULTIHORIZON_PATH, target_keys=list(manifest['output_labels']), sequence_len=seq, batch_size=1024, amp_enabled=False, amp_dtype='bfloat16')
dir_probs, hold_probs, conf_probs = np.split(probs,3,axis=1)
cal=walk['overall']['calibration']
p=apply_bucket_calibration(dir_probs[:,2], cal)
precision=json.loads(resolve_precision_gate_path(manifest).read_text())
meta=load_meta_gate(resolve_meta_gate_path(manifest))
ctx=np.load(GATE_CONTEXT_PATH, mmap_mode='r')[seq-1:seq-1+usable].astype(np.float32)
_,_,gate=_combined_gate_scores(probs, precision, meta, context_features=ctx)
th=walk['optimized_thresholds']
dec=th['decision_threshold']; cf=th['confidence_floor']; hth=th['hold_threshold']; gth=th['gate_threshold']
print(json.dumps({
 'rows': int(len(p)),
 'prob_ge_dec': int((p>=dec).sum()),
 'prob_le_short': int((p<=1-dec).sum()),
 'conf_ge_floor': int((conf_probs[:,2]>=cf).sum()),
 'hold_lt_thresh': int((hold_probs[:,2]<hth).sum()),
 'gate_ge_thresh': int((gate>=gth).sum()),
 'final_longs': int(((p>=dec)&(conf_probs[:,2]>=cf)&(hold_probs[:,2]<hth)&(gate>=gth)).sum()),
 'final_shorts': int(((p<=1-dec)&(conf_probs[:,2]>=cf)&(hold_probs[:,2]<hth)&(gate>=gth)).sum()),
 'gate_stats': {'min': float(gate.min()), 'max': float(gate.max()), 'mean': float(gate.mean())},
 'hold_stats': {'min': float(hold_probs[:,2].min()), 'max': float(hold_probs[:,2].max()), 'mean': float(hold_probs[:,2].mean())},
 'conf_stats': {'min': float(conf_probs[:,2].min()), 'max': float(conf_probs[:,2].max()), 'mean': float(conf_probs[:,2].mean())},
 'prob_stats': {'min': float(p.min()), 'max': float(p.max()), 'mean': float(p.mean())}
}, indent=2))