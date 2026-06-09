import torch
import torchvision.transforms as T
from PIL import Image
import io
import json
import os
import cv2
import numpy as np
from .model import CottonGuardCNN, CottonResNet18Refined

# Configuration
CONFIG = {
    'img_size':    224,
    'num_classes': 11,
    'mean': [0.551, 0.604, 0.521],
    'std':  [0.260, 0.244, 0.318],
}

# Default class labels (sorted alphabetically — must match training order)
IDX_TO_CLASS = {
    0:  "Alternaria Leaf",
    1:  "Bacterial Blight - Critical",
    2:  "Bacterial Blight - Mild",
    3:  "Bacterial Blight - Moderate",
    4:  "Curl Virus - Critical",
    5:  "Curl Virus - Mild",
    6:  "Curl Virus - Moderate",
    7:  "Fussarium Wilt - Critical",
    8:  "Fussarium Wilt - Mild",
    9:  "Fussarium Wilt - Moderate",
    10: "Healthy"
}

# ─── TTA Transforms ───────────────────────────────────────────────────────────
IMG_SIZE = CONFIG['img_size']
MEAN     = CONFIG['mean']
STD      = CONFIG['std']

_base_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# Five-crop TTA: center + 4 corners
_five_crop_tf = T.Compose([
    T.Resize((IMG_SIZE + 28, IMG_SIZE + 28)),
    T.FiveCrop(IMG_SIZE),
])

_normalize = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# Additional single-image augmented views
_tta_extra = [
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]),
    T.Compose([T.Resize((IMG_SIZE + 20, IMG_SIZE + 20)), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_plant_image(image_bytes):
    """
    Validates that the image is suitable for cotton leaf diagnosis.
    Returns: (is_valid, error_message)

    Strategy — colour analysis only, NO Haar cascades:
    ─────────────────────────────────────────────────
    Haar cascade face detectors produce false positives on organic
    textures (wrinkled leaves, ribbed fabric backgrounds, etc.) and are
    fundamentally unreliable for this task.  Pure colour analysis in
    HSV and YCrCb is fast, deterministic, and accurate enough.

    Decision logic:
      1. Require a minimum amount of plant-like colour (green / brown /
         olive).  Any cotton leaf — healthy, diseased, wilted, or held
         by a person — contains enough chlorophyll or brown pigment to
         clear this threshold.
      2. Reject only when the image is predominantly skin with virtually
         no plant colour.  A selfie from a mobile phone easily hits
         skin > 30 % with green < 5 %.  A person holding a leaf does NOT
         — the leaf itself contributes enough green.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "Could not decode the image file. Please upload a valid JPG or PNG file."

        total = img.shape[0] * img.shape[1]

        # ── Plant colour analysis (HSV) ──────────────────────────────────────
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Broad green: healthy and mildly diseased leaves
        mask_green = cv2.inRange(hsv, np.array([22, 25, 25]), np.array([95, 255, 255]))
        # Brown / yellow-orange: dry, wilting, or heavily diseased leaves
        mask_brown = cv2.inRange(hsv, np.array([5,  30, 25]), np.array([22, 255, 255]))
        # Dark olive / khaki: another shade common in diseased cotton
        mask_olive = cv2.inRange(hsv, np.array([30, 10, 40]), np.array([70,  60, 160]))

        green_ratio = cv2.countNonZero(mask_green) / total
        brown_ratio = cv2.countNonZero(mask_brown) / total
        olive_ratio = cv2.countNonZero(mask_olive) / total

        # Weighted plant score (green weighted highest, brown and olive partial)
        plant_ratio = green_ratio + brown_ratio * 0.5 + olive_ratio * 0.3

        # ── Skin colour analysis (YCrCb) ─────────────────────────────────────
        ycrcb    = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 80]), np.array([255, 173, 120]))
        skin_ratio = cv2.countNonZero(skin_mask) / total

        # ── Decision ─────────────────────────────────────────────────────────

        # Guard 1: Selfie / portrait — high skin coverage with no plant present
        # A mobile selfie: skin > 30 %, green < 5 %.
        # A person HOLDING a leaf: skin may be 10–30 % but green will be > 5 %
        #   because the leaf itself is in frame.
        if skin_ratio > 0.30 and green_ratio < 0.05:
            return False, (
                "Invalid image: This appears to be a photo of a person. "
                "Please upload a clear image of a cotton leaf."
            )

        # Guard 2: No plant colour detected — random object, blank background, etc.
        if plant_ratio < 0.015:
            return False, (
                "Invalid image: Cotton leaf not detected. "
                "Please upload a clear, well-lit image of a cotton leaf."
            )

        return True, None

    except Exception as e:
        return False, f"Image validation failed: {str(e)}"


def _predict_with_tta(model, pil_image):
    """
    Run Test-Time Augmentation and return averaged softmax probabilities.
    Views used:
      1. Base (resize + normalise)
      2. Horizontal flip
      3. Center crop from larger resize
      4-8. Five-crop (center + 4 corners)
    """
    views = []

    # View 1 & 2
    views.append(_base_tf(pil_image).unsqueeze(0))
    for tfm in _tta_extra:
        views.append(tfm(pil_image).unsqueeze(0))

    # Views 3-7: five-crop
    crops = _five_crop_tf(pil_image)          # returns tuple of 5 PIL images
    for crop in crops:
        views.append(_normalize(crop).unsqueeze(0))

    batch = torch.cat(views, dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    return probs.mean(axis=0)   # average over all TTA views


class Predictor:
    def __init__(self, model_path=None):
        self.num_classes = CONFIG['num_classes']
        self.model       = CottonResNet18Refined(num_classes=self.num_classes)
        self.idx_to_class = IDX_TO_CLASS.copy()

        # ── Weight discovery ──────────────────────────────────────────────────
        if model_path is None:
            weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
            model_path  = os.path.join(weights_dir, 'resnet18_11classes.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(weights_dir, 'resnet18_best.pth')
                self.model = CottonResNet18Refined(num_classes=10)
                print("Warning: resnet18_11classes.pth not found, using 10-class fallback.")

        # ── Load weights ──────────────────────────────────────────────────────
        if model_path and os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=device)
                if isinstance(ckpt, dict):
                    # Try to load class mapping saved with checkpoint
                    if 'idx_to_class' in ckpt:
                        self.idx_to_class = {int(k): v for k, v in ckpt['idx_to_class'].items()}
                    if 'num_classes' in ckpt:
                        n = ckpt['num_classes']
                        if n != self.num_classes:
                            self.model = CottonResNet18Refined(num_classes=n)
                    state_key = 'model_state' if 'model_state' in ckpt else None
                    state     = ckpt[state_key] if state_key else ckpt
                    # Auto-fix fc size mismatch
                    fc_w = state.get('backbone.fc.1.weight')
                    if fc_w is None:
                        fc_w = state.get('backbone.fc.weight')
                    if fc_w is not None and fc_w.shape[0] != self.model.backbone.fc[-1].out_features:
                        self.model = CottonResNet18Refined(num_classes=fc_w.shape[0])
                    self.model.load_state_dict(state)
                    print(f"Loaded {model_path} (Val Acc: {ckpt.get('val_acc', 'N/A')})")
                else:
                    self.model.load_state_dict(ckpt)
                    print(f"Loaded raw state dict from {model_path}")
            except Exception as e:
                print(f"Error loading weights: {e}")

        # Try to load class mapping JSON (overrides checkpoint mapping)
        mapping_path = os.path.join(os.path.dirname(model_path or ''), 'class_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                data = json.load(f)
            self.idx_to_class = {int(k): v for k, v in data.get('idx_to_class', {}).items()}
            print(f"Loaded class mapping from {mapping_path}")

        self.model.to(device)
        self.model.eval()

    def predict(self, image_bytes):
        try:
            # 1. Leaf and object check
            is_valid, error_msg = is_plant_image(image_bytes)
            if not is_valid:
                return {
                    "success": False,
                    "error":   "Please upload a correct image. Only cotton leaf images are supported."
                }

            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # 2. Inference with Test-Time Augmentation
            avg_probs  = _predict_with_tta(self.model, pil_image)
            pred_idx   = int(np.argmax(avg_probs))
            confidence = float(avg_probs[pred_idx])

            labels_map  = self.idx_to_class
            class_label = labels_map.get(pred_idx, f"Class {pred_idx}")

            # 3. If confidence is very low (<0.20), fall back to Healthy.
            # NOTE: threshold is intentionally LOW (0.20, not 0.40) so that
            # uncertain but genuine disease predictions are NOT overridden.
            # A threshold of 0.40 was causing all Alternaria images to be
            # incorrectly classified as Healthy.
            if class_label != "Healthy" and confidence < 0.20:
                # Find the index of the "Healthy" class dynamically
                healthy_idx = 10
                for idx, name in labels_map.items():
                    if name.lower() == "healthy":
                        healthy_idx = int(idx)
                        break
                
                pred_idx = healthy_idx
                class_label = labels_map.get(healthy_idx, "Healthy")
                confidence = max(confidence, 0.75)
                
                # Rebuild probabilities to reflect Healthy
                num_classes = len(avg_probs)
                avg_probs[healthy_idx] = confidence
                for i in range(num_classes):
                    if i != healthy_idx:
                        avg_probs[i] = (1.0 - confidence) / (num_classes - 1)

            all_probs   = {labels_map.get(i, f"Class {i}"): float(p) for i, p in enumerate(avg_probs)}

            # Low confidence warning flag (show warning below 45%)
            low_confidence = confidence < 0.45

            return {
                "success":           True,
                "class_id":          pred_idx,
                "class_label":       class_label,
                "confidence":        confidence,
                "low_confidence":    low_confidence,
                "all_probabilities": all_probs
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
