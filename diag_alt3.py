import os, sys, torch
sys.path.insert(0, r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torchvision.transforms as T
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy\backend\weights\resnet18_11classes.pth'

ckpt = torch.load(weights_path, map_location=device)
from backend.model import CottonResNet18Refined
model = CottonResNet18Refined(num_classes=ckpt['num_classes'])
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()

MEAN = ckpt.get('mean', [0.551, 0.604, 0.521])
STD  = ckpt.get('std',  [0.260, 0.244, 0.318])
tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
idx_to_class = {int(k): v for k,v in ckpt['idx_to_class'].items()}

# Test on aug_Alternaria_Leaf folder (the actual TRAINING source)
alt_folder = r'E:\DL AGRICULTURE\aug_Alternaria_Leaf'
images = [f for f in os.listdir(alt_folder) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))][:15]

print('=== Testing on aug_Alternaria_Leaf (TRAINING source) ===')
results = {}
for img_name in images:
    img = Image.open(os.path.join(alt_folder, img_name)).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    pred_label = idx_to_class[pred_idx]
    results[pred_label] = results.get(pred_label, 0) + 1
    print(f'  {img_name[:40]:40s} -> {pred_label:25s} (alt={probs[0]:.3f}, conf={probs[pred_idx]:.3f})')

print()
print('Prediction distribution (from training data!):')
for k,v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')

# Check model FC layer weights for Alternaria class
print()
fc_weight = None
for name, param in model.named_parameters():
    if 'fc.1.weight' in name or 'fc.weight' in name:
        fc_weight = param.data
        print(f'FC layer: {name}, shape: {param.shape}')
        # Check norm of Alternaria row (idx 0)
        print(f'  Alternaria (row 0) L2 norm: {fc_weight[0].norm().item():.4f}')
        print(f'  Healthy    (row 10) L2 norm: {fc_weight[10].norm().item():.4f}')
        norms = [(idx_to_class[i], fc_weight[i].norm().item()) for i in range(fc_weight.shape[0])]
        norms.sort(key=lambda x: -x[1])
        print('  FC weight norms by class:')
        for lbl, n in norms:
            print(f'    {lbl:35s}: {n:.4f}')
        break
