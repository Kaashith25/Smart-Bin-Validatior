import torch
import clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LENGTH = 70

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

def safe_text(text):
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH]
    return text

def encode_texts(texts):
    texts = [safe_text(t) for t in texts]
    with torch.no_grad():
        tokens = clip.tokenize(texts).to(DEVICE)
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features

def encode_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_feature = clip_model.encode_image(img_tensor)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    return image_feature
