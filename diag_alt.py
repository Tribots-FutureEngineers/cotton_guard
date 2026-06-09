import torch
import torchvision.transforms as T
from PIL import Image
import json, os, sys
sys.path.insert(0, r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from backend.model import CottonResNet18Refined

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy\backend\weights\resnet18_11classes.pth'

ckpt = torch.load(weights_path, map_location=device)
print('Keys in checkpoint:', list(ckpt.keys()))
print('num_classes:', ckpt.get('num_classes'))
print('val_acc:', ckpt.get('val_acc'))
print('val_f1_macro:', ckpt.get('val_f1_macro'))
print('idx_to_class:', ckpt.get('idx_to_class'))

# Load model
model = CottonResNet18Refined(num_classes=ckpt['num_classes'])
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()

MEAN = ckpt.get('mean', [0.551, 0.604, 0.521])
STD  = ckpt.get('std',  [0.260, 0.244, 0.318])
tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

idx_to_class = {int(k): v for k,v in ckpt['idx_to_class'].items()}

# Test on first 5 Alternaria images from deployment folder
alt_folder = r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy\Alternaria Leaf Spot Cotton'
if not os.path.exists(alt_folder):
    print('Alternaria folder not found at:', alt_folder)
else:
    images = [f for f in os.listdir(alt_folder) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))][:10]
    print(f'\nTesting on {len(images)} Alternaria images:')
    for img_name in images:
        img = Image.open(os.path.join(alt_folder, img_name)).convert('RGB')
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        print(f'  {img_name[:40]:40s} -> {idx_to_class[pred_idx]:30s} (conf={probs[pred_idx]:.3f}, alternaria_conf={probs[0]:.3f})')
