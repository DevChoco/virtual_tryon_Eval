import os
from PIL import Image
import torch
import open_clip
from torchvision import transforms
import pandas as pd

clothes_dir = "clothes"
synthesized_dir = "synthesized_images"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

with torch.no_grad():
    text_tokens = tokenizer(["A person wearing a short sleeve", "A person wearing a long sleeve"]).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

text_map = {
    "short": 0,
    "long": 1
}

def get_sleeve_label(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ text_feats.T).squeeze()
        pred_idx = torch.argmax(sims).item()
        return "short" if pred_idx == 0 else "long"

def get_clipscore(image_path, text_index):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

        sim = (img_feat @ text_feats[text_index].unsqueeze(0).T).item()
        return sim
        
results = []

for fname in os.listdir(clothes_dir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base_name = fname  # 예: s0012.jpg
    clothes_path = os.path.join(clothes_dir, base_name)
    synth_name = f"gen_{base_name[:8]}_{base_name}"
    synth_path = os.path.join(synthesized_dir, synth_name)

    if not os.path.exists(synth_path):
        print(f"[경고] 합성 이미지 없음: {synth_path}")
        continue

    try:
        # 옷 이미지에서 라벨 추출
        true_label = get_sleeve_label(clothes_path)
        true_index = text_map[true_label]
        wrong_index = 1 - true_index

        # 합성 이미지의 CLIPScore
        correct_score = get_clipscore(synth_path, true_index)
        wrong_score = get_clipscore(synth_path, wrong_index)

        results.append({
            "filename": base_name,
            "true_label": true_label,
            "correct_score": round(correct_score, 4),
            "wrong_score": round(wrong_score, 4),
            "match": correct_score > wrong_score
        })

    except Exception as e:
        print(f"[에러] {fname} 처리 중 오류 발생: {e}")

df = pd.DataFrame(results)
df.to_csv("clip_sleeve_eval.csv", index=False)
print("평가 완료: clip_sleeve_eval.csv 저장됨.")
