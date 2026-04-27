import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO

def overlay_yolo_on_mask(img_path, mask_path, yolo_model):
    """
    Overlays YOLO detections from a source image onto a segmentation mask.
    """
    if not os.path.exists(img_path):
        print(f"Error: Image path '{img_path}' does not exist.")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask path '{mask_path}' does not exist.")
        return

    # Load images
    base_img = cv2.imread(img_path)
    mask_img = cv2.imread(mask_path)

    if base_img is None or mask_img is None:
        print("Error: Could not read images.")
        return

    # Run YOLO detection
    results = yolo_model(base_img)
    
    # Process results and overlay on mask
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())
            label = f"{yolo_model.names[cls]} {conf:.2f}"

            # Draw rectangle on the mask image (or a copy of it)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(mask_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(mask_img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return mask_img

def main():
    parser = argparse.ArgumentParser(description="Nexus Vision: Agentic Fusion Pipeline")
    parser.add_argument("--img", type=str, required=True, help="Path to the source color image")
    parser.add_argument("--mask", type=str, required=True, help="Path to the segmentation mask")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--output", type=str, default="fused_output.png", help="Path to save output")
    
    args = parser.parse_args()

    print(f"🚀 Initializing Fusion Pipeline...")
    yolo_model = YOLO(args.model)
    
    fused_img = overlay_yolo_on_mask(args.img, args.mask, yolo_model)
    
    if fused_img is not None:
        cv2.imwrite(args.output, fused_img)
        print(f"✅ Fusion Complete! Saved to: {args.output}")
        
        # Optionally show the result
        cv2.imshow("Fused Agentic Vision", fused_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
