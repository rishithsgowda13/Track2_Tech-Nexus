import torch
import numpy as np
import cv2
import os
import argparse
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# ============================================================================
# Model Definition (DINOv2 + ConvNeXt Head)
# ============================================================================
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=7, padding=3), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ============================================================================
# Visual Palette
# ============================================================================
color_palette = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235],
], dtype=np.uint8)

def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(color_palette)):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

def add_hud_label(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    overlay = img.copy()
    cv2.rectangle(overlay, (x-10, y-h-10), (x+w+10, y+10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(img, (x-10, y+10), (x+w+10, y+10), (0, 255, 0), 2)

# ============================================================================
# Core Processing Logic
# ============================================================================
def process_image(img_path, output_path, seg_model_path, yolo_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Image
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print(f"Error: Could not read {img_path}")
        return

    # Resize to standard panel size
    panel_w, panel_h = 480, 270
    img_orig = cv2.resize(img_orig, (panel_w, panel_h))
    
    # 1. Patch Perspective
    patch_view = img_orig.copy()
    grid_spacing = 14 * 2 # Standard DINOv2 patch size scaled
    overlay = patch_view.copy()
    for x in range(0, panel_w, grid_spacing): cv2.line(overlay, (x, 0), (x, panel_h), (255, 255, 0), 1)
    for y in range(0, panel_h, grid_spacing): cv2.line(overlay, (0, y), (panel_w, y), (255, 255, 0), 1)
    cv2.addWeighted(overlay, 0.3, patch_view, 0.7, 0, patch_view)
    patch_view = cv2.resize(patch_view, (panel_w//14, panel_h//14), interpolation=cv2.INTER_NEAREST)
    patch_view = cv2.resize(patch_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

    # 2. Load Models and Run Segmentation
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14").eval().to(device)
    classifier = SegmentationHeadConvNeXt(384, 10, 476//14, 266//14).eval().to(device)
    if os.path.exists(seg_model_path):
        classifier.load_state_dict(torch.load(seg_model_path, map_location=device))
    
    img_pil = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((266, 476)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
        out = F.interpolate(classifier(feat), size=(panel_h, panel_w), mode="bilinear", align_corners=False)
        mask = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.uint8)
    seg_color = cv2.cvtColor(mask_to_color(mask), cv2.COLOR_RGB2BGR)

    # 3. YOLO Detection
    yolo = YOLO(yolo_model_path)
    fused = seg_color.copy()
    yolo_results = yolo(img_orig, verbose=False)
    for r in yolo_results:
        for box in r.boxes:
            conf = box.conf[0].cpu().item()
            label_name = yolo.names[int(box.cls[0])]
            if conf > 0.5 and label_name not in ["donut", "pizza", "cake"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(fused, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(fused, label_name.upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Apply HUD and Combine
    add_hud_label(img_orig, "I. INPUT", (20, 25))
    add_hud_label(patch_view, "II. PATCHES", (20, 25))
    add_hud_label(seg_color, "III. TERRAIN", (20, 25))
    add_hud_label(fused, "IV. FUSION", (20, 25))
    
    padding = 10
    final = np.hstack((img_orig, 
                       np.full((panel_h, padding, 3), 20, dtype=np.uint8),
                       patch_view, 
                       np.full((panel_h, padding, 3), 20, dtype=np.uint8),
                       seg_color, 
                       np.full((panel_h, padding, 3), 20, dtype=np.uint8),
                       fused))
    
    cv2.imwrite(output_path, final)
    print(f"✅ Processed: {output_path}")
    
    # Auto-open the image for the judge (Windows)
    try:
        os.startfile(output_path)
        print(f"👁️ Opening {output_path}...")
    except Exception as e:
        print(f"Could not auto-open image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_showcase.png")
    parser.add_argument("--seg_model", type=str, default="segmentation_head.pth")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    args = parser.parse_args()
    process_image(args.input, args.output, args.seg_model, args.yolo_model)
