import os
import torch
from hf import push_to_hub
from config import TrainingConfig
from train import (
    apply_lora_adapters, 
    load_and_prepare_dataset, 
    load_model_and_tokenizer, 
    train_model
)

os.makedirs(TrainingConfig.OUTPUT_DIR, exist_ok=True)

model, tokenizer = load_model_and_tokenizer()
model = apply_lora_adapters(model)
dataset = load_and_prepare_dataset(tokenizer)
trainer = train_model(model, tokenizer, dataset)

# First save the LoRA model for backup
lora_output_dir = os.path.join(TrainingConfig.OUTPUT_DIR, "lora_model")
model.save_pretrained(lora_output_dir)
print(f"LoRA model saved to {lora_output_dir}")

# Merge LoRA weights with base model
print("Merging LoRA weights with base model...")
merged_model = model.merge_and_unload()

# Save merged model and tokenizer
merged_output_dir = os.path.join(TrainingConfig.OUTPUT_DIR, "merged_model")
os.makedirs(merged_output_dir, exist_ok=True)
merged_model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)
print(f"Merged model and tokenizer saved to {merged_output_dir}")

# Clear GPU memory
del merged_model
torch.cuda.empty_cache()

# push finetuned model to HF Hub
push_to_hub()