"""
Train Reviewer Agent
Fine-tunes Phi-2 for quality classification using OpenAssistant dataset
"""

import os
import sys
import json
import torch
from transformers import TrainingArguments, Trainer

# Import from HuggingFace BEFORE adding local path
from datasets import Dataset as HFDataset
import logging

# Add parent directory to path AFTER importing datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, 
    DATASET_CONFIG, PATH_CONFIG
)
from utils.model_loader import load_base_model, prepare_model_for_training
from peft import get_peft_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATH_CONFIG.logs_dir, 'train_reviewer.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_training_data():
    """
    Load prepared training datasets
    
    Returns:
        dict: Train and validation datasets
    """
    logger.info("Loading training data...")
    
    data_dir = DATASET_CONFIG.reviewer_output_dir
    
    # Load train data
    train_path = os.path.join(data_dir, 'train.json')
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Load validation data
    val_path = os.path.join(data_dir, 'validation.json')
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    logger.info(f"Loaded {len(train_data)} training samples")
    logger.info(f"Loaded {len(val_data)} validation samples")
    
    return {
        'train': HFDataset.from_list(train_data),
        'validation': HFDataset.from_list(val_data)
    }


def tokenize_function(examples, tokenizer):
    """
    Tokenize examples
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer
    
    Returns:
        dict: Tokenized examples
    """
    # Tokenize prompts
    tokenized = tokenizer(
        examples['prompt'],
        truncation=True,
        max_length=MODEL_CONFIG.max_length,
        padding='max_length',
        return_tensors=None
    )
    
    # Set labels (same as input_ids for causal LM)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def train_reviewer_agent():
    """
    Main training function for Reviewer Agent
    """
    logger.info("=" * 60)
    logger.info("Training Reviewer Agent")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(PATH_CONFIG.reviewer_adapter_path, exist_ok=True)
    os.makedirs(PATH_CONFIG.logs_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info("\n📥 Loading base model...")
    model, tokenizer = load_base_model()
    
    # Prepare model for training
    model = prepare_model_for_training(model)
    
    # Add LoRA adapters
    logger.info("\n🔧 Adding LoRA adapters...")
    peft_config = LORA_CONFIG.to_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Load datasets
    logger.info("\n📚 Loading datasets...")
    datasets = load_training_data()
    
    # Tokenize datasets
    logger.info("🔄 Tokenizing datasets...")
    tokenized_datasets = {
        'train': datasets['train'].map(
            lambda x: tokenize_function(x, tokenizer),
            batched=False,
            remove_columns=datasets['train'].column_names
        ),
        'validation': datasets['validation'].map(
            lambda x: tokenize_function(x, tokenizer),
            batched=False,
            remove_columns=datasets['validation'].column_names
        )
    }
    
    # Setup training arguments
    logger.info("\n⚙️  Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=PATH_CONFIG.reviewer_adapter_path,
        **TRAINING_CONFIG.to_dict(),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create trainer
    logger.info("\n🏋️  Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )
    
    # Train
    logger.info("\n🚀 Starting training...")
    logger.info(f"Total epochs: {TRAINING_CONFIG.num_train_epochs}")
    logger.info(f"Batch size: {TRAINING_CONFIG.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {TRAINING_CONFIG.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {TRAINING_CONFIG.per_device_train_batch_size * TRAINING_CONFIG.gradient_accumulation_steps}")
    
    train_result = trainer.train()
    
    # Save model
    logger.info("\n💾 Saving model...")
    trainer.save_model(PATH_CONFIG.reviewer_adapter_path)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics_path = os.path.join(PATH_CONFIG.reviewer_adapter_path, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training complete!")
    logger.info(f"Model saved to: {PATH_CONFIG.reviewer_adapter_path}")
    logger.info("=" * 60)
    
    # Print final metrics
    logger.info("\n📊 Training Metrics:")
    for key, value in metrics.items():
        logger.info(f"   {key}: {value}")
    
    return True


if __name__ == "__main__":
    try:
        success = train_reviewer_agent()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
