from transformers import AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model
from config import Config

def setup_dora_model():
    base_model = AutoModelForVision2Seq.from_pretrained(
        Config.MODEL_ID,
        device_map=Config.DEVICE
    )
    
    peft_config = LoraConfig(
        r=Config.LORA_RANK,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
        use_dora=Config.USE_DORA
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    return model

if __name__ == "__main__":
    model = setup_dora_model()
