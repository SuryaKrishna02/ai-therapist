import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import create_repo
from config import HFConfig, TrainingConfig

def push_to_hub():
    """Push the merged model and tokenizer to HuggingFace Hub"""
    try:
        # Create repository if it doesn't exist
        create_repo(HFConfig.REPO_ID, private=False, token=HFConfig.TOKEN, exist_ok=True)
        print(f"Repository {HFConfig.REPO_ID} is ready")

        # Load the merged model and tokenizer
        merged_model_path = os.path.join(TrainingConfig.OUTPUT_DIR, "merged_model")
        print("Loading merged model and tokenizer...")
        
        model = AutoModelForCausalLM.from_pretrained(merged_model_path)
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        
        # Push to Hub
        print(f"Pushing model and tokenizer to {HFConfig.REPO_ID}...")
        model.push_to_hub(
            HFConfig.REPO_ID,
            token=HFConfig.TOKEN,
            commit_message=HFConfig.COMMIT_MESSAGE
        )
        tokenizer.push_to_hub(
            HFConfig.REPO_ID,
            token=HFConfig.TOKEN
        )
        
        print("Successfully pushed model and tokenizer to HuggingFace Hub!")
        print(f"Model is available at: https://huggingface.co/{HFConfig.REPO_ID}")
        
    except Exception as e:
        print(f"Error pushing to HuggingFace Hub: {str(e)}")
        raise