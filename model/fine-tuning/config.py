from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    MODEL_NAME: str = "unsloth/Qwen2.5-7B-Instruct"
    MAX_SEQ_LENGTH: int = 2048
    DTYPE = None
    LOAD_IN_4BIT: bool = True
    RANK: int = 16
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.0

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 1
    BATCH_SIZE: int = 2
    GRAD_ACC_STEPS: int = 4
    DATASET_PATH: str = "../dataset/train.jsonl"
    OUTPUT_DIR: str = "outputs"

@dataclass
class HFConfig:
    """Configuration for HuggingFace"""
    REPO_ID: str = "SuryaKrishna02/Qwen2.5-7B-therapy"  # Change this
    TOKEN: str = "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"  # Change this to your HF token
    COMMIT_MESSAGE: str = "Add fine-tuned model with LoRA adapters"