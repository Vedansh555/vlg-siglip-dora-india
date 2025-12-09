import torch

class Config:
    MODEL_ID = "google/siglip-base-patch16-224"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # DoRA Settings
    LORA_RANK = 16
    LORA_ALPHA = 32
    USE_DORA = True
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    EPOCHS = 5
