# 🚀 Nexus Vision: Judge's Presentation Commands

Use these pre-formatted commands to demonstrate the AI's situational awareness on different terrain samples. Just copy and paste them into your terminal.

---

### 🏜️ 1. Training Dataset Samples (Complex Terrain)

**Sample 015 (Desert Ridge)**
```powershell
python nexus_vision_process.py --input "Dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images/cc0000015.png" --output "result_015.png"
```

**Sample 018 (Arid Landscape)**
```powershell
python nexus_vision_process.py --input "Dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images/cc0000018.png" --output "result_018.png"
```

**Sample 025 (Rugged Terrain)**
```powershell
python nexus_vision_process.py --input "Dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images/cc0000025.png" --output "result_025.png"
```

---

### 🧪 2. Test Dataset Samples (Unseen Environments)

**Sample 062 (Offroad Path)**
```powershell
python nexus_vision_process.py --input "TestImages/Offroad_Segmentation_testImages/Color_Images/0000062.png" --output "test_062.png"
```

**Sample 065 (Clear Horizon)**
```powershell
python nexus_vision_process.py --input "TestImages/Offroad_Segmentation_testImages/Color_Images/0000065.png" --output "test_065.png"
```

**Sample 069 (Diverse Vegetation)**
```powershell
python nexus_vision_process.py --input "TestImages/Offroad_Segmentation_testImages/Color_Images/0000069.png" --output "test_069.png"
```

---

### 🛠️ 3. Generic Command (For ANY image)

If the judge gives you a custom image (e.g., `new_sample.jpg`), use this template:
```powershell
python nexus_vision_process.py --input "new_sample.jpg" --output "new_result.png"
```

---

### 🎥 4. Live Pathfinding Demo (Webcam)
To show the AI navigating in real-time, run:
```powershell
python fusion_vision.py
```

> [!TIP]
> After running any command, you can open the output file (e.g., `result_015.png`) to show the **4-Panel AI Brain View** (Original -> Patches -> Terrain -> Fusion).
