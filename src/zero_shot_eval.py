from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import pandas as pd

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).cuda()
processor = CLIPProcessor.from_pretrained(model_name)

df = pd.read_csv("data/test.csv")  # labels + image paths
labels = sorted(df["label"].unique())
prompts = [f"a photo of {l}" for l in labels]

def encode_text(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)

text_emb = encode_text(prompts)

top1 = 0
top5 = 0

for _, row in df.iterrows():
    img = Image.open(row["image"]).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to("cuda")

    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    sims = (img_emb @ text_emb.T).squeeze()
    sorted_idx = sims.argsort(descending=True)

    true = labels.index(row["label"])
    if sorted_idx[0].item() == true: top1 += 1
    if true in sorted_idx[:5].tolist(): top5 += 1

print("Top-1 Accuracy:", top1/len(df)*100)
print("Top-5 Accuracy:", top5/len(df)*100)
