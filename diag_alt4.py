import os, sys, torch
sys.path.insert(0, r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torchvision.transforms as T
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy\backend\weights\resnet18_11classes.pth', map_location=device)
from backend.model import CottonResNet18Refined
model = CottonResNet18Refined(num_classes=ckpt['num_classes'])
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()

MEAN = ckpt.get('mean', [0.551, 0.604, 0.521])
STD  = ckpt.get('std',  [0.260, 0.244, 0.318])
tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
idx_to_class = {int(k): v for k,v in ckpt['idx_to_class'].items()}

# KEY TEST: compare pixel statistics of training vs extra alternaria images
from PIL import Image
import numpy as np

def get_stats(folder, n=20):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png','.webp')) and os.path.isfile(os.path.join(folder,f))][:n]
    means, stds = [], []
    for f in files:
        img = np.array(Image.open(os.path.join(folder, f)).convert('RGB').resize((224,224))) / 255.0
        means.append(img.mean(axis=(0,1)))
        stds.append(img.std(axis=(0,1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)

print('=== Image Statistics Comparison ===')
train_mean, train_std = get_stats(r'E:\DL AGRICULTURE\aug_Alternaria_Leaf')
print(f'aug_Alternaria_Leaf (training): mean={train_mean.round(3)}, std={train_std.round(3)}')

extra_mean, extra_std = get_stats(r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy\Alternaria Leaf Spot Cotton')
print(f'Alternaria Leaf Spot Cotton (extra): mean={extra_mean.round(3)}, std={extra_std.round(3)}')

# Also test: run model on ALL 23 extra images and tabulate
print()
print('=== Model predictions on ALL 23 extra Alternaria images ===')
folder = r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy\Alternaria Leaf Spot Cotton'
all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
dist = {}
for img_name in all_files:
    img = Image.open(os.path.join(folder, img_name)).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    pred = idx_to_class[pred_idx]
    alt_conf = probs[0]
    dist[pred] = dist.get(pred, 0) + 1
    # flag those that go to Healthy after inference threshold
    after_threshold = 'Healthy (overridden)' if (pred != 'Healthy' and probs[pred_idx] < 0.40) else pred
    print(f'  {img_name[:40]:40s} -> raw={pred:25s} (alt={alt_conf:.3f}, conf={probs[pred_idx]:.3f}) | after_thresh={after_threshold}')

print()
print('Raw prediction distribution:')
for k,v in sorted(dist.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')
