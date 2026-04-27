from ultralytics import YOLO
import cv2
import numpy as np

print("🚀 Initializing Agentic Fusion: Advanced Pathfinding Mode...")
# Using YOLOv8-seg for combined detection and segmentation
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("✅ Advanced Navigation Ready! Scanning for clear paths...")
print("Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    annotated_frame = frame.copy()
    
    # 1. Detect Obstacles
    results = model(frame, verbose=False, conf=0.3)
    
    # Create a 1D map of blocked horizontal space (0 = clear, 1 = blocked)
    blocked_map = np.zeros(w, dtype=np.uint8)
    
    for r in results:
        # Plot segmentation masks for techy look
        if r.masks is not None:
            annotated_frame = r.plot(boxes=False)
            
        for box in r.boxes:
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())
            label_name = model.names[cls]
            
            # Filter out non-obstacle objects
            if label_name not in ["donut", "pizza", "cake"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Only care about obstacles in the lower 2/3 of the screen (ground level)
                if y2 > h // 3:
                    # Mark this horizontal range as blocked
                    # Add a buffer (padding) around obstacles for safety
                    buffer = 30 
                    start = max(0, x1 - buffer)
                    end = min(w, x2 + buffer)
                    blocked_map[start:end] = 1
                    
                    # Draw red box for obstacles
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"BLOCK: {label_name}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 2. Find Clear Gaps (Looking for the "Shortest/Nearest" Path)
    clear_indices = np.where(blocked_map == 0)[0]
    
    best_gap_center = w // 2 # Default to straight ahead
    found_path = False
    
    if len(clear_indices) > 0:
        # Group contiguous clear indices into gaps
        gaps = []
        current_gap = [clear_indices[0]]
        for i in range(1, len(clear_indices)):
            if clear_indices[i] == clear_indices[i-1] + 1:
                current_gap.append(clear_indices[i])
            else:
                gaps.append(current_gap)
                current_gap = [clear_indices[i]]
        gaps.append(current_gap)
        
        # Filter gaps that are wide enough to pass through
        min_gap_width = w // 4
        valid_gaps = [g for g in gaps if len(g) > min_gap_width]
        
        if valid_gaps:
            # Shortest Path Logic: Find gap center closest to the current center (w//2)
            centers = [ (g[0] + g[-1]) // 2 for g in valid_gaps ]
            best_gap_idx = np.argmin([abs(c - (w//2)) for c in centers])
            best_gap_center = centers[best_gap_idx]
            found_path = True

    # 3. Draw Navigation Guidance
    status_text = "NO CLEAR PATH FOUND"
    status_color = (0, 0, 255)
    
    if found_path:
        # Logic to decide movement
        center_offset = best_gap_center - (w // 2)
        
        if abs(center_offset) < 60: # Threshold for "straight"
            status_text = "PATH CLEAR: PROCEED STRAIGHT"
            status_color = (0, 255, 0)
        else:
            direction = "LEFT" if center_offset < 0 else "RIGHT"
            status_text = f"OBSTACLE! STEER {direction}"
            status_color = (0, 255, 255)

        # Draw a Dynamic Directional Arrow
        arrow_base = (w // 2, h - 40)
        arrow_tip = (best_gap_center, h // 2 + 50)
        cv2.arrowedLine(annotated_frame, arrow_base, arrow_tip, (0, 255, 0), 8, tipLength=0.2)
        
        # Draw "Goal" Indicator
        cv2.circle(annotated_frame, arrow_tip, 10, (0, 255, 0), -1)
        cv2.putText(annotated_frame, "OPTIMAL PATH", (best_gap_center - 60, h // 2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 4. Display Modern Overlay
    cv2.rectangle(annotated_frame, (0, 0), (w, 70), (20, 20, 20), -1) # Dark header
    cv2.putText(annotated_frame, f"SYSTEM: {status_text}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    # Progress Bar for "Blocked" zones at bottom
    for i in range(w // 10):
        color = (0, 0, 255) if blocked_map[i*10] == 1 else (0, 255, 0)
        cv2.rectangle(annotated_frame, (i*10, h-10), (i*10 + 8, h-2), color, -1)

    cv2.imshow("Nexus Vision: Pathfinding Advisor", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
