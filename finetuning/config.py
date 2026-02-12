"""
Configuration for Fine-Tuning Reviewer and Fact-Checker Agents
Optimized for GTX 1650 (4GB VRAM) using Phi-2 with QLoRA
"""

from peft import LoraConfig
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Base model configuration"""
    base_model_name: str = "microsoft/phi-2"
    max_length: int = 256
    device_map: str = "auto"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration for QLoRA fine-tuning"""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Phi-2 attention modules
            self.target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    
    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig"""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            target_modules=self.target_modules
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for GTX 1650"""
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = True
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TrainingArguments"""
        return {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "fp16": self.fp16,
            "optim": self.optim,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
        }


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    # Dataset sizes (optimized for GTX 1650)
    train_size: int = 4000
    val_size: int = 500
    test_size: int = 500
    
    # Reviewer dataset (OpenAssistant)
    reviewer_dataset_name: str = "OpenAssistant/oasst1"
    reviewer_output_dir: str = "finetuning/data_preparation/reviewer_data"
    
    # Fact-checker dataset (FEVER)
    factchecker_dataset_name: str = "fever"
    factchecker_config: str = "v1.0"
    factchecker_output_dir: str = "finetuning/data_preparation/factchecker_data"
    
    # Label mappings
    reviewer_labels: Dict[str, int] = None
    factchecker_labels: Dict[str, int] = None
    
    def __post_init__(self):
        if self.reviewer_labels is None:
            self.reviewer_labels = {
                "weak": 0,
                "acceptable": 1,
                "good": 2
            }
        if self.factchecker_labels is None:
            self.factchecker_labels = {
                "REFUTES": 0,
                "NOT_ENOUGH_INFO": 1,
                "SUPPORTS": 2
            }


@dataclass
class PathConfig:
    """Path configuration for models and adapters"""
    # Base directories
    finetuning_dir: str = "finetuning"
    models_dir: str = "finetuning/models"
    
    # Adapter paths
    reviewer_adapter_path: str = "finetuning/models/phi2-reviewer-lora"
    factchecker_adapter_path: str = "finetuning/models/phi2-factchecker-lora"
    
    # Evaluation results
    eval_results_dir: str = "finetuning/results"
    
    # Logs
    logs_dir: str = "finetuning/logs"


# Global configuration instances
MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
DATASET_CONFIG = DatasetConfig()
PATH_CONFIG = PathConfig()


# Instruction templates
REVIEWER_INSTRUCTION = "Evaluate the quality of the following research paragraph. Classify it as 'good', 'acceptable', or 'weak' based on clarity, coherence, evidence support, and academic rigor."

FACTCHECKER_INSTRUCTION = "Verify the following claim using the provided evidence. Classify as 'SUPPORTS' if the evidence confirms the claim, 'REFUTES' if it contradicts the claim, or 'NOT_ENOUGH_INFO' if there is insufficient evidence."


def get_prompt_template(instruction: str, input_text: str, output: str = None) -> str:
    """
    Generate prompt template for training
    
    Args:
        instruction: Task instruction
        input_text: Input text to process
        output: Expected output (for training)
    
    Returns:
        Formatted prompt string
    """
    if output is not None:
        # Training format
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        # Inference format
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
