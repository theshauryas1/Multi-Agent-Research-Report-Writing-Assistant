"""
Training Configuration - QLoRA settings optimized for RTX 2080 Ti.
Provides safe defaults for 11GB VRAM with batch size adjustments.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for QLoRA fine-tuning.
    
    Optimized for RTX 2080 Ti (11GB VRAM) with Mistral 7B.
    """
    
    # Model settings
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_type: str = "mistral"
    
    # LoRA configuration
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Quantization (4-bit for VRAM efficiency)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 2  # Safe for 11GB VRAM
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    
    # Context length
    max_seq_length: int = 2048  # Balanced for VRAM
    
    # Optimizer
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler: str = "cosine"
    
    # Precision
    fp16: bool = True
    bf16: bool = False  # RTX 2080 Ti doesn't support bf16 well
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    report_to: str = "none"  # or "tensorboard"
    
    # Directories
    output_dir: str = "outputs/training"
    cache_dir: str = "cache/models"
    
    # Dataset
    dataset_path: Optional[str] = None
    train_split: float = 0.9
    
    # Evaluation
    eval_steps: int = 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            k: v if not isinstance(v, list) else v.copy()
            for k, v in self.__dict__.items()
        }
    
    def get_lora_config(self):
        """Get LoRA configuration for PEFT."""
        try:
            from peft import LoraConfig, TaskType
            
            return LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        except ImportError:
            raise ImportError("peft is required. Install with: pip install peft")
    
    def get_bnb_config(self):
        """Get BitsAndBytes configuration for quantization."""
        if not self.use_4bit:
            return None
        
        try:
            import torch
            from transformers import BitsAndBytesConfig
            
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.use_double_quant,
            )
        except ImportError:
            raise ImportError(
                "bitsandbytes is required. Install with: pip install bitsandbytes"
            )
    
    def get_training_args(self):
        """Get Transformers TrainingArguments."""
        try:
            from transformers import TrainingArguments
            
            return TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio,
                max_grad_norm=self.max_grad_norm,
                fp16=self.fp16,
                bf16=self.bf16,
                save_strategy=self.save_strategy,
                save_steps=self.save_steps,
                save_total_limit=self.save_total_limit,
                logging_steps=self.logging_steps,
                evaluation_strategy="steps" if self.eval_steps else "no",
                eval_steps=self.eval_steps if self.eval_steps else None,
                report_to=self.report_to,
                optim=self.optimizer,
                lr_scheduler_type=self.lr_scheduler,
                gradient_checkpointing=True,  # Save VRAM
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except ImportError:
            raise ImportError("transformers is required")


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_config_for_vram(vram_gb: int) -> TrainingConfig:
    """
    Get training configuration optimized for specific VRAM.
    
    Args:
        vram_gb: Available VRAM in GB
        
    Returns:
        Optimized TrainingConfig
    """
    config = TrainingConfig()
    
    if vram_gb >= 24:
        # High-end GPU (RTX 3090, A5000, etc.)
        config.batch_size = 4
        config.max_seq_length = 4096
        config.gradient_accumulation_steps = 4
    elif vram_gb >= 16:
        # Mid-range (RTX 4080, etc.)
        config.batch_size = 3
        config.max_seq_length = 2048
        config.gradient_accumulation_steps = 6
    elif vram_gb >= 11:
        # RTX 2080 Ti, RTX 3060, etc.
        config.batch_size = 2
        config.max_seq_length = 2048
        config.gradient_accumulation_steps = 8
    elif vram_gb >= 8:
        # RTX 3070, etc.
        config.batch_size = 1
        config.max_seq_length = 1024
        config.gradient_accumulation_steps = 16
    else:
        # Low VRAM
        config.batch_size = 1
        config.max_seq_length = 512
        config.gradient_accumulation_steps = 32
        config.lora_r = 8
    
    return config


if __name__ == "__main__":
    # Test configuration
    print("Testing training configuration...")
    
    config = get_default_config()
    
    print(f"\nDefault config for RTX 2080 Ti:")
    print(f"  Model: {config.base_model}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  4-bit quantization: {config.use_4bit}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Test VRAM-specific configs
    for vram in [8, 11, 16, 24]:
        cfg = get_config_for_vram(vram)
        print(f"\n{vram}GB VRAM config:")
        print(f"  Batch: {cfg.batch_size}, Seq len: {cfg.max_seq_length}")
