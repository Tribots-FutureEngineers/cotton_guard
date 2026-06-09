"""
CottonGuard Diagnostic Script — Phase 1 Audit
=============================================
Runs all 5 audits WITHOUT modifying any code or weights.
"""

import sys
import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
BACK   = os.path.join(BASE, "backend")
WDIR   = os.path.join(BACK, "weights")
SDIR   = os.path.join(BACK, "samples")

sys.path.insert(0, BASE)

# Add backend to path so we can import model
sys.path.insert(0, BACK)

WEIGHTS = {
    "resnet18_11classes.pth": os.path.join(WDIR, "resnet18_11classes.pth"),
    "resnet18_best.pth":      os.path.join(WDIR, "resnet18_best.pth"),
    "best_model.pth":         os.path.join(WDIR, "best_model.pth"),
}
CLASS_MAP_PATH = os.path.join(WDIR, "class_mapping.json")

HARDCODED_IDX_TO_CLASS = {
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

device = torch.device("cpu")

SEP = "=" * 70

def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ─── STEP 1: Checkpoint Inspection ───────────────────────────────────────────
banner("STEP 1 · CHECKPOINT METADATA AUDIT")

for fname, fpath in WEIGHTS.items():
    if not os.path.exists(fpath):
        print(f"  [MISSING] {fname}")
        continue
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"\n  File: {fname}  ({size_mb:.1f} MB)")
    try:
        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            print(f"  Checkpoint keys: {keys}")
            for key in ["val_acc", "epoch", "train_acc", "train_loss", "val_loss",
                        "num_classes", "class_weights", "loss_fn", "optimizer",
                        "history", "idx_to_class", "class_to_idx",
                        "best_val_acc", "config"]:
                if key in ckpt:
                    val = ckpt[key]
                    if isinstance(val, (dict, list)) and len(str(val)) > 200:
                        print(f"  {key}: [long value, abbreviated] {str(val)[:200]} ...")
                    else:
                        print(f"  {key}: {val}")
            # Show fc layer shape to determine num_classes
            state_key = "model_state" if "model_state" in ckpt else None
            state = ckpt[state_key] if state_key else ckpt
            for k, v in state.items():
                if "fc" in k and "weight" in k:
                    print(f"  State key '{k}' shape: {tuple(v.shape)}  → num_classes = {v.shape[0]}")
        else:
            # raw state dict
            print(f"  Raw state dict (no metadata wrapper)")
            for k, v in ckpt.items():
                if "fc" in k and "weight" in k:
                    print(f"  State key '{k}' shape: {tuple(v.shape)}  → num_classes = {v.shape[0]}")
    except Exception as e:
        print(f"  ERROR loading {fname}: {e}")

# ─── STEP 2: Class Mapping Audit ─────────────────────────────────────────────
banner("STEP 2 · CLASS MAPPING AUDIT")

print("\n  [A] Hardcoded IDX_TO_CLASS in inference.py:")
for idx, name in HARDCODED_IDX_TO_CLASS.items():
    print(f"      {idx:>2}: {name}")

print("\n  [B] class_mapping.json on disk:")
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH) as f:
        cm = json.load(f)
    for idx, name in cm.get("idx_to_class", {}).items():
        print(f"      {idx:>2}: {name}")
    print("\n  class_to_idx from JSON:")
    for name, idx in cm.get("class_to_idx", {}).items():
        print(f"      {idx:>2}: {name}")
else:
    print("  [MISSING] class_mapping.json not found")

# Cross-check hardcoded vs JSON
print("\n  [C] Cross-check hardcoded vs JSON:")
json_map = {int(k): v for k, v in cm.get("idx_to_class", {}).items()}
mismatches = []
for idx in range(max(len(HARDCODED_IDX_TO_CLASS), len(json_map))):
    hc = HARDCODED_IDX_TO_CLASS.get(idx, "MISSING")
    jm = json_map.get(idx, "MISSING")
    match = "✓" if hc == jm else "✗ MISMATCH"
    if hc != jm:
        mismatches.append(idx)
    print(f"      idx {idx:>2}: hardcoded='{hc}' | json='{jm}' [{match}]")
if not mismatches:
    print("  → All class labels match between hardcoded and JSON.")
else:
    print(f"  → MISMATCHES found at indices: {mismatches}")

# ─── STEP 3: Training History (from checkpoint) ───────────────────────────────
banner("STEP 3 · TRAINING HISTORY AUDIT (from checkpoint)")

primary_ckpt_path = WEIGHTS["resnet18_11classes.pth"]
if os.path.exists(primary_ckpt_path):
    try:
        ckpt = torch.load(primary_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            # Look for history
            if "history" in ckpt:
                history = ckpt["history"]
                print(f"\n  Training history found. Type: {type(history)}")
                if isinstance(history, dict):
                    for epoch_key, epoch_data in list(history.items())[:5]:
                        print(f"  Epoch {epoch_key}: {epoch_data}")
                elif isinstance(history, list):
                    print(f"  Total epochs recorded: {len(history)}")
                    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>9}")
                    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*9}")
                    for i, h in enumerate(history):
                        if isinstance(h, dict):
                            tl = h.get("train_loss", h.get("loss", "N/A"))
                            vl = h.get("val_loss", "N/A")
                            va = h.get("val_acc", h.get("accuracy", "N/A"))
                            tl_s = f"{tl:.4f}" if isinstance(tl, float) else str(tl)
                            vl_s = f"{vl:.4f}" if isinstance(vl, float) else str(vl)
                            va_s = f"{va:.4f}" if isinstance(va, float) else str(va)
                            print(f"  {i+1:>6}  {tl_s:>12}  {vl_s:>10}  {va_s:>9}")
            else:
                print("\n  No 'history' key found in checkpoint.")
                print(f"  Available metadata keys: {[k for k in ckpt.keys() if not k in ('model_state',)]}")

            # Check for class weights
            if "class_weights" in ckpt:
                cw = ckpt["class_weights"]
                print(f"\n  Class weights stored in checkpoint: {cw}")
            else:
                print("\n  No 'class_weights' key found → Loss function likely used no class weighting.")

            # Check config
            if "config" in ckpt:
                print(f"\n  Training config: {ckpt['config']}")
        else:
            print("  Checkpoint is a raw state dict — no training metadata available.")
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print("  Primary checkpoint not found.")

# ─── STEP 4: Inference Audit on Sample Images ────────────────────────────────
banner("STEP 4 · INFERENCE AUDIT (Sample Images)")

try:
    from model import CottonResNet18Refined

    # Load model
    model = CottonResNet18Refined(num_classes=11)
    ckpt_path = WEIGHTS["resnet18_11classes.pth"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state_key = "model_state" if "model_state" in ckpt else None
        state = ckpt[state_key] if state_key else ckpt
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    print("  Model loaded successfully.\n")

    # Build transforms
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.551, 0.604, 0.521], std=[0.260, 0.244, 0.318]),
    ])

    # Run on available sample images
    sample_images = []
    for fname in os.listdir(SDIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            sample_images.append((fname, os.path.join(SDIR, fname)))

    if not sample_images:
        print("  No sample images found in backend/samples/")
    else:
        print(f"  {'Image File':<30} {'Actual Class (from filename)':<35} {'Top Prediction':<35} {'Confidence':>12}")
        print(f"  {'-'*30} {'-'*35} {'-'*35} {'-'*12}")

        all_probs_list = []
        for fname, fpath in sample_images:
            img = Image.open(fpath).convert("RGB")
            tensor = tf(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            pred_label = HARDCODED_IDX_TO_CLASS.get(pred_idx, f"Class {pred_idx}")
            actual_guess = fname.replace(".png","").replace(".jpg","").replace(".jpeg","")
            print(f"  {fname:<30} {actual_guess:<35} {pred_label:<35} {conf:>12.1%}")
            all_probs_list.append((fname, probs))

        # Print full probability distribution per image
        print(f"\n  ── Full probability distributions ──")
        for fname, probs in all_probs_list:
            print(f"\n  Image: {fname}")
            sorted_indices = np.argsort(probs)[::-1]
            for rank, idx in enumerate(sorted_indices):
                label = HARDCODED_IDX_TO_CLASS.get(idx, f"Class {idx}")
                print(f"    #{rank+1:>2}  {label:<35} {probs[idx]:>7.2%}")

except Exception as e:
    print(f"  ERROR during inference: {e}")
    import traceback
    traceback.print_exc()

# ─── STEP 5: Logit / Softmax Analysis ────────────────────────────────────────
banner("STEP 5 · RAW LOGIT ANALYSIS (Bias Detection)")

try:
    from model import CottonResNet18Refined

    model2 = CottonResNet18Refined(num_classes=11)
    ckpt = torch.load(WEIGHTS["resnet18_11classes.pth"], map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state_key = "model_state" if "model_state" in ckpt else None
        state = ckpt[state_key] if state_key else ckpt
    else:
        state = ckpt
    model2.load_state_dict(state)
    model2.eval()

    # Inspect the final FC layer weights and biases
    fc_layer = None
    for name, module in model2.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 11:
            fc_layer = module
            fc_name = name

    if fc_layer is not None:
        weights_fc = fc_layer.weight.data.numpy()  # shape: (11, in_features)
        biases_fc  = fc_layer.bias.data.numpy()    # shape: (11,)

        print(f"\n  Final FC layer: '{fc_name}'")
        print(f"  Weight shape: {weights_fc.shape}")
        print(f"\n  ── Per-class Bias (raw logit offset before any input) ──")
        print(f"  {'Class':>3}  {'Label':<35} {'Bias':>10}  {'Weight L2 Norm':>15}")
        print(f"  {'-'*3}  {'-'*35} {'-'*10}  {'-'*15}")
        for i in range(11):
            label = HARDCODED_IDX_TO_CLASS.get(i, f"Class {i}")
            bias_val = biases_fc[i]
            w_norm = float(np.linalg.norm(weights_fc[i]))
            print(f"  {i:>3}  {label:<35} {bias_val:>10.4f}  {w_norm:>15.4f}")

        # Show if Fussarium classes have higher biases
        fuss_indices = [7, 8, 9]
        non_fuss = [i for i in range(11) if i not in fuss_indices]
        avg_fuss_bias = np.mean(biases_fc[fuss_indices])
        avg_non_fuss_bias = np.mean(biases_fc[non_fuss])
        print(f"\n  Average bias — Fussarium classes (7,8,9):     {avg_fuss_bias:+.4f}")
        print(f"  Average bias — Non-Fussarium classes:          {avg_non_fuss_bias:+.4f}")
        if avg_fuss_bias > avg_non_fuss_bias + 0.2:
            print("  ⚠️  Fussarium classes have significantly higher bias → strong model bias toward Fussarium.")
        else:
            print("  Bias values do not clearly favour Fussarium at the weight level.")

        # Soft test: push a zero-mean noise image through
        print(f"\n  ── Response to zero-mean random noise (10 trials, averaged) ──")
        noise_preds = np.zeros(11)
        for _ in range(10):
            noise = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                logits = model2(noise)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
            noise_preds += probs
        noise_preds /= 10
        sorted_n = np.argsort(noise_preds)[::-1]
        for rank, idx in enumerate(sorted_n[:5]):
            label = HARDCODED_IDX_TO_CLASS.get(idx, f"Class {idx}")
            print(f"    #{rank+1}  {label:<35} {noise_preds[idx]:.2%}")

    else:
        print("  Could not find final FC layer.")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# ─── STEP 6: best_model.pth separate inspection ──────────────────────────────
banner("STEP 6 · best_model.pth INSPECTION (likely different architecture)")

bm_path = WEIGHTS["best_model.pth"]
if os.path.exists(bm_path):
    size_mb = os.path.getsize(bm_path) / 1e6
    print(f"\n  File size: {size_mb:.1f} MB")
    try:
        ckpt = torch.load(bm_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            print(f"  Keys: {list(ckpt.keys())}")
            for key in ["val_acc", "epoch", "num_classes", "class_weights", "loss_fn", "config", "idx_to_class"]:
                if key in ckpt:
                    print(f"  {key}: {ckpt[key]}")
            state_key = "model_state" if "model_state" in ckpt else None
            state = ckpt[state_key] if state_key else ckpt
            for k, v in state.items():
                if "weight" in k and ("fc" in k or "classifier" in k or "head" in k):
                    print(f"  Layer '{k}' shape: {tuple(v.shape)}")
        else:
            for k, v in ckpt.items():
                if "weight" in k and ("fc" in k or "classifier" in k or "head" in k):
                    print(f"  Layer '{k}' shape: {tuple(v.shape)}")
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print("  best_model.pth not found.")

banner("DIAGNOSIS COMPLETE")
print("""
  Summary of checks performed:
  1. Checkpoint metadata (keys, val_acc, epochs, class weights, training config)
  2. Class label mapping consistency (hardcoded vs JSON)
  3. Training history (per-epoch loss/acc if stored in checkpoint)
  4. Inference on available sample images with full probability distribution
  5. Raw FC-layer bias and weight analysis (detects learned class bias)
  6. best_model.pth architecture inspection

  See output above for findings.
""")
