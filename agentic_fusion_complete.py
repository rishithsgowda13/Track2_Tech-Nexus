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
# Model: Segmentation Head (Must match training)
# ============================================================================
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
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
# Visualization Utils
# ============================================================================
color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(color_palette)):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

# ============================================================================
# Main Fusion Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Nexus Vision: Complete Agentic Fusion")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--seg_model", type=str, default="segmentation_head.pth", help="Path to custom segmentation model")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--output", type=str, default="fusion_result.png", help="Output path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load DINOv2 Backbone
    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone.eval().to(device)

    # 2. Load Custom Segmentation Head
    print(f"Loading Segmentation Head from {args.seg_model}...")
    w, h = 476, 266 # Standard size used in training
    if not os.path.exists(args.seg_model):
        print(f"WARNING: {args.seg_model} not found! Will use random weights for demonstration.")
        classifier = SegmentationHeadConvNeXt(in_channels=384, out_channels=10, tokenW=w//14, tokenH=h//14)
    else:
        classifier = SegmentationHeadConvNeXt(in_channels=384, out_channels=10, tokenW=w//14, tokenH=h//14)
        classifier.load_state_dict(torch.load(args.seg_model, map_location=device))
    
    classifier.eval().to(device)

    # 3. Load YOLO Model
    print(f"Loading YOLO from {args.yolo_model}...")
    yolo = YOLO(args.yolo_model)

    # 4. Process Image
    img_orig = cv2.imread(args.img)
    if img_orig is None:
        print(f"Error: Could not read image {args.img}")
        return
    
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 5. Run Segmentation
    with torch.no_grad():
        features = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
        logits = classifier(features)
        seg_output = F.interpolate(logits, size=(img_orig.shape[0], img_orig.shape[1]), mode="bilinear", align_corners=False)
        pred_mask = torch.argmax(seg_output[0], dim=0).cpu().numpy().astype(np.uint8)
    
    # Convert mask to color
    seg_color = mask_to_color(pred_mask)
    seg_color_bgr = cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR)

    # 6. Run YOLO Detection
    yolo_results = yolo(img_orig)
    
    # 7. Fuse: Overlay YOLO boxes on Segmentation Mask
    fused_img = seg_color_bgr.copy()
    for r in yolo_results:
        for box in r.boxes:
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())
            label_name = yolo.names[cls]

            # DONUT FILTER: Ignore food and low confidence in offroad settings
            if conf > 0.5 and label_name not in ["donut", "pizza", "cake", "sandwich", "broccoli"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                label = f"{label_name} {conf:.2f}"

                # Draw on fused image
                cv2.rectangle(fused_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(fused_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 8. Save and Show
    cv2.imwrite(args.output, fused_img)
    print(f"✅ Agentic Fusion Complete! Result saved to {args.output}")
    
    # Side-by-side comparison
    combined = np.hstack((img_orig, seg_color_bgr, fused_img))
    cv2.imshow("Nexus Vision: Original | Segmentation | Fused", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
