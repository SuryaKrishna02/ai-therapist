from typing import Tuple
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from config import ModelConfig, TrainingConfig
from unsloth import FastLanguageModel, is_bfloat16_supported


def load_model_and_tokenizer() -> Tuple:
    """Load the base model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ModelConfig.MODEL_NAME,
        max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
        dtype=ModelConfig.DTYPE,
        load_in_4bit=ModelConfig.LOAD_IN_4BIT,
    )
    return model, tokenizer

def apply_lora_adapters(model):
    """Apply LoRA adapters to the model"""
    model = FastLanguageModel.get_peft_model(
        model,
        r=ModelConfig.RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=ModelConfig.LORA_ALPHA,
        lora_dropout=ModelConfig.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model

def format_example(example: dict, eos_token: str) -> dict:
    """Format dataset example"""
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output_text = example["output"].strip()
    text = f"<<SYS>>{instruction}<<SYS>>\n\n<<CLIENT>>{input_text}<<CLIENT>>\n\n<<THERAPIST>>{output_text}<<THERAPIST>>{eos_token}"
    return {"text": text}

def load_and_prepare_dataset(tokenizer):
    """Load and prepare the dataset"""
    dataset = load_dataset("json", data_files=TrainingConfig.DATASET_PATH, split="train")
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    dataset = dataset.map(
        lambda ex: format_example(ex, eos_token), 
        batched=False
    )
    return dataset

def train_model(model, tokenizer, dataset):
    """Train the model"""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
            gradient_accumulation_steps=TrainingConfig.GRAD_ACC_STEPS,
            warmup_steps=5,
            num_train_epochs=TrainingConfig.NUM_EPOCHS,
            learning_rate=TrainingConfig.LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=TrainingConfig.OUTPUT_DIR,
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")
    return trainer