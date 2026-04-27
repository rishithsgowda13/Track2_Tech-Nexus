import torch
import numpy as np
import cv2
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# ============================================================================
# Presentation Config
# ============================================================================
IMG_LIST = [
    "Dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images/cc0000015.png",
    "Dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images/cc0000205.png"
]
SEG_MODEL = "segmentation_head.pth"
YOLO_MODEL = "yolov8n.pt"
OUTPUT_FILE = "Nexus_Vision_Presentation.png"

# ============================================================================
# Model Definitions
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

color_palette = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235],
], dtype=np.uint8)

def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(color_palette)):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

# ============================================================================
# High-Tech Visualization Utils
# ============================================================================
def add_hud_label(img, text, pos):
    """Add a professional HUD-style label to a panel."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x-10, y-h-10), (x+w+10, y+10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    # Text and decorative line
    cv2.putText(img, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(img, (x-10, y+10), (x+w+10, y+10), (0, 255, 0), 2)

def create_showcase():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing Premium Showcase on {device}...")

    # Load models
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14").eval().to(device)
    w_feat, h_feat = 476, 266
    classifier = SegmentationHeadConvNeXt(384, 10, w_feat//14, h_feat//14).eval().to(device)
    if os.path.exists(SEG_MODEL):
        classifier.load_state_dict(torch.load(SEG_MODEL, map_location=device))
        print("✅ Custom Segmentation Model Loaded.")
    
    yolo = YOLO(YOLO_MODEL)
    
    panel_w, panel_h = 480, 270 # 16:9 Aspect Ratio
    padding = 20
    rows = []
    
    transform = transforms.Compose([
        transforms.Resize((h_feat, w_feat)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for img_path in IMG_LIST:
        if not os.path.exists(img_path): continue
        print(f"Processing {img_path}...")
        
        # 1. Original
        img_orig = cv2.imread(img_path)
        img_orig = cv2.resize(img_orig, (panel_w, panel_h))
        
        # 2. AI Patch Perspective (High-Tech Grid)
        patch_view = img_orig.copy()
        # Create a subtle cyan grid
        grid_color = (255, 255, 0) # Cyan
        grid_spacing = int(panel_w / (w_feat / 14))
        overlay = patch_view.copy()
        for x in range(0, panel_w, grid_spacing):
            cv2.line(overlay, (x, 0), (x, panel_h), grid_color, 1)
        for y in range(0, panel_h, grid_spacing):
            cv2.line(overlay, (0, y), (panel_w, y), grid_color, 1)
        cv2.addWeighted(overlay, 0.3, patch_view, 0.7, 0, patch_view)
        # Pixelation effect
        patch_view = cv2.resize(patch_view, (w_feat//14, h_feat//14), interpolation=cv2.INTER_NEAREST)
        patch_view = cv2.resize(patch_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

        # 3. Segmentation
        img_pil = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
            logits = classifier(feat)
            out = F.interpolate(logits, size=(panel_h, panel_w), mode="bilinear", align_corners=False)
            mask = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.uint8)
        seg_color = cv2.cvtColor(mask_to_color(mask), cv2.COLOR_RGB2BGR)
        
        # 4. Fused (Clean Detection)
        fused = seg_color.copy()
        yolo_results = yolo(img_orig, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                label_name = yolo.names[cls]
                if conf > 0.5 and label_name not in ["donut", "pizza", "cake", "sandwich"]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(fused, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(fused, label_name.upper(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Apply HUD Labels
        add_hud_label(img_orig, "I. INPUT STREAM", (20, 30))
        add_hud_label(patch_view, "II. PATCH ANALYSIS", (20, 30))
        add_hud_label(seg_color, "III. TERRAIN MAP", (20, 30))
        add_hud_label(fused, "IV. AGENTIC FUSION", (20, 30))

        # Combine Panels with Padding
        row_img = np.full((panel_h, panel_w * 4 + padding * 3, 3), 40, dtype=np.uint8) # Dark background
        row_img[:, 0:panel_w] = img_orig
        row_img[:, panel_w + padding : panel_w*2 + padding] = patch_view
        row_img[:, panel_w*2 + padding*2 : panel_w*3 + padding*2] = seg_color
        row_img[:, panel_w*3 + padding*3 : panel_w*4 + padding*3] = fused
        rows.append(row_img)

    # Vertical Stack with Padding
    final_canvas = np.full((panel_h * len(rows) + padding * (len(rows)+1), panel_w * 4 + padding * 5, 3), 20, dtype=np.uint8)
    for i, row in enumerate(rows):
        y_start = padding + i * (panel_h + padding)
        final_canvas[y_start : y_start + panel_h, padding : padding + row.shape[1]] = row

    # Add Main Title
    cv2.putText(final_canvas, "NEXUS VISION: MULTI-STAGE SITUATIONAL AWARENESS", (padding, padding - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_FILE, final_canvas)
    print(f"\n✨ Premium Showcase Created: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_showcase()
