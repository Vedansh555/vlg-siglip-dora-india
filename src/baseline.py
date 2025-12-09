import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from config import Config

def run_baseline(image_path, candidate_labels):
    model = AutoModel.from_pretrained(Config.MODEL_ID).to(Config.DEVICE)
    processor = AutoProcessor.from_pretrained(Config.MODEL_ID)

    image = Image.open(image_path)
    
    inputs = processor(
        text=candidate_labels, 
        images=image, 
        padding="max_length", 
        return_tensors="pt"
    ).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image)

    print("\n--- Baseline Results ---")
    for i, label in enumerate(candidate_labels):
        print(f"{label}: {probs[0][i].item():.4f}")

if __name__ == "__main__":
    run_baseline("data/indi_bench_examples/test.jpg", ["A Saree", "A Dress", "A T-Shirt"])
