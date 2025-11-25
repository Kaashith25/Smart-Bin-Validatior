import os
import re
from PIL import Image
import torch
import clip
from ultralytics import YOLO
from collections import Counter

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BALANCED THRESHOLD:
# 0.20 filters out "Books as Indiana Jones" (Hallucinations)
# but lets "Messy Database Text" pass through.
SIMILARITY_THRESHOLD = 0.20

print(f"Initializing Zero-Shot Inference on {DEVICE}...")

# ================= LOAD MODELS =================
yolo = YOLO("yolov8x-seg.pt")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# ================= HELPER: PROMPT ENSEMBLING =================
def clean_text_data(text):
    text = re.sub(r'[^a-zA-Z0-9 \-]', '', text)
    return text

def get_ensemble_features(text):
    # Simple clean up
    text = clean_text_data(text)
    short_text = " ".join(text.split()[:15])[:70]
    
    prompts = [
        f"A photo of {short_text}",
        f"The product {short_text}",
        f"{short_text}"
    ]
    
    tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        mean_feature = features.mean(dim=0, keepdim=True)
        mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
    
    return mean_feature

# ================= CORE ANALYSIS =================
def analyze_image(img_path, target_descriptions):
    
    # 1. YOLO DETECTION (High Sensitivity)
    # conf=0.01 finds objects in corners, replacing the need for manual tiling
    results = yolo(img_path, conf=0.01, iou=0.5, agnostic_nms=True, verbose=False)[0]
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    
    crops = []
    
    # --- STRATEGY: YOLO + 1 BACKUP ---
    # We removed Left/Right tiles to fix the "Found 12" bug.
    
    # 1. Center Crop (Only 1 backup)
    # Captures the main pile if YOLO fails completely
    crops.append(clip_preprocess(img.crop((width*0.1, height*0.1, width*0.9, height*0.9))))

    # 2. YOLO Crops (The primary counter)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        w, h = x2 - x1, y2 - y1
        
        # Filter Noise
        if w < width * 0.05 or h < height * 0.05: continue 
        if w > width * 0.95 or h > height * 0.95: continue 
        
        pad = 25
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad); y2 = min(height, y2 + pad)
        crops.append(clip_preprocess(img.crop((x1, y1, x2, y2))))

    if not crops: return {} 

    # 2. PREPARE TEXT FEATURES
    search_keys = []
    text_features_list = []

    # A. Encode User Descriptions
    for desc in target_descriptions:
        feat = get_ensemble_features(desc)
        text_features_list.append(feat)
        search_keys.append(desc)
            
    # B. BACKGROUND CLASS
    # Filters out empty shelves/walls
    bg_prompts = "empty shelf empty bin yellow tape wall floor"
    bg_feat = get_ensemble_features(bg_prompts)
    text_features_list.append(bg_feat)
    search_keys.append("BACKGROUND")

    # 3. CLIP INFERENCE
    with torch.no_grad():
        text_features_matrix = torch.cat(text_features_list)
        image_batch = torch.stack(crops).to(DEVICE)
        image_features = clip_model.encode_image(image_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = image_features @ text_features_matrix.T

    # 4. COUNT MATCHES
    detected_counts = Counter()

    for i in range(len(similarity)):
        scores = similarity[i]
        top_idx = scores.argmax().item()
        max_score = scores[top_idx].item()
        
        pred_key = search_keys[top_idx]
        
        # LOGIC: Matches if > 0.20 AND not Background
        if pred_key != "BACKGROUND" and max_score > SIMILARITY_THRESHOLD:
            detected_counts[pred_key] += 1

    return detected_counts