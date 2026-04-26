import json
lines = open("outputs/v24/diffusion_v2_phase06_log.jsonl").readlines()
for l in lines[-15:]:
    d = json.loads(l)
    print(f"epoch={d['epoch']}, total={d['train_total']:.4f}, val_loss={d.get('val_loss','?')}, best={d.get('best',False)}")