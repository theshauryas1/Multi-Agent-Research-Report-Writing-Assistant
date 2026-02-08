"""
QLoRA Trainer - Fine-tuning script for Mistral 7B.
Optimized for RTX 2080 Ti with efficient memory usage.
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from .config import TrainingConfig, get_default_config
from .dataset import DatasetLoader, create_training_dataset


class Trainer:
    """
    QLoRA fine-tuning trainer for Mistral 7B.
    
    Handles model loading, LoRA configuration, and training loop.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or get_default_config()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup(self) -> None:
        """
        Set up model, tokenizer, and LoRA configuration.
        """
        print(f"Setting up trainer for {self.config.base_model}...")
        
        self._load_tokenizer()
        self._load_model()
        self._apply_lora()
        
        print("✓ Trainer setup complete")
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.tokenizer.padding_side = "right"
            
            print(f"✓ Loaded tokenizer: {self.config.base_model}")
            
        except ImportError:
            raise ImportError("transformers is required. Install with: pip install transformers")
    
    def _load_model(self) -> None:
        """Load the base model with quantization."""
        try:
            import torch
            from transformers import AutoModelForCausalLM
            
            bnb_config = self.config.get_bnb_config()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Prepare for training
            self.model.config.use_cache = False
            self.model.config.pretraining_tp = 1
            
            print(f"✓ Loaded model with {'4-bit' if self.config.use_4bit else 'full precision'} quantization")
            
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}")
    
    def _apply_lora(self) -> None:
        """Apply LoRA configuration to model."""
        try:
            from peft import get_peft_model, prepare_model_for_kbit_training
            
            # Prepare model for k-bit training
            if self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Apply LoRA
            lora_config = self.config.get_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable, total = self._count_parameters()
            print(f"✓ Applied LoRA: {trainable:,} / {total:,} trainable params ({100*trainable/total:.2f}%)")
            
        except ImportError:
            raise ImportError("peft is required. Install with: pip install peft")
    
    def _count_parameters(self) -> tuple:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total
    
    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Args:
            train_data: Training dataset (list of {"text": ...} dicts)
            val_data: Validation dataset
            
        Returns:
            Training metrics
        """
        try:
            from datasets import Dataset
            from trl import SFTTrainer
            
            # Create HuggingFace datasets
            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data) if val_data else None
            
            print(f"Training on {len(train_dataset)} samples")
            if val_dataset:
                print(f"Validation on {len(val_dataset)} samples")
            
            # Create output directory
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Set up trainer
            training_args = self.config.get_training_args()
            
            self.trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                packing=False,
            )
            
            # Train
            print("\n" + "="*50)
            print("Starting training...")
            print("="*50 + "\n")
            
            train_result = self.trainer.train()
            
            # Save the final model
            self.save()
            
            print("\n" + "="*50)
            print("Training complete!")
            print("="*50)
            
            return {
                "train_loss": train_result.training_loss,
                "metrics": train_result.metrics,
            }
            
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Install with: pip install trl datasets")
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the fine-tuned model.
        
        Args:
            path: Save path (uses config.output_dir if not provided)
        """
        save_path = path or self.config.output_dir
        
        # Save LoRA adapter
        adapter_path = os.path.join(save_path, "adapter")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        print(f"✓ Saved LoRA adapter to {adapter_path}")
    
    def load_adapter(self, adapter_path: str) -> None:
        """
        Load a saved LoRA adapter.
        
        Args:
            adapter_path: Path to saved adapter
        """
        try:
            from peft import PeftModel
            
            # Load base model if not already loaded
            if self.model is None:
                self._load_tokenizer()
                self._load_model()
            
            # Load adapter
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
            print(f"✓ Loaded adapter from {adapter_path}")
            
        except ImportError:
            raise ImportError("peft is required")
    
    def merge_and_save(self, output_path: str) -> None:
        """
        Merge LoRA adapter with base model and save.
        
        Args:
            output_path: Path to save merged model
        """
        print("Merging adapter with base model...")
        
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print(f"✓ Saved merged model to {output_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        import torch
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated


def train_model(
    data_path: str,
    output_dir: str = "outputs/training",
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        data_path: Path to training data
        output_dir: Output directory
        config: Training configuration
        
    Returns:
        Training results
    """
    # Set up config
    if config is None:
        config = get_default_config()
    config.output_dir = output_dir
    
    # Load data
    train_data, val_data = create_training_dataset(data_path)
    
    # Create trainer
    trainer = Trainer(config)
    trainer.setup()
    
    # Train
    results = trainer.train(train_data, val_data)
    
    return results


if __name__ == "__main__":
    print("QLoRA Trainer for Mistral 7B")
    print("=" * 50)
    
    # This is a demo - actual training requires data
    config = get_default_config()
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.base_model}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    
    print(f"\nTo train:")
    print("  1. Prepare training data in JSONL format")
    print("  2. Run: python -m training.train --data path/to/data.jsonl")
    
    # Check if we can load the model
    print(f"\nChecking dependencies...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("  ✗ PyTorch not found")
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError:
        print("  ✗ Transformers not found")
    
    try:
        import peft
        print(f"  ✓ PEFT {peft.__version__}")
    except ImportError:
        print("  ✗ PEFT not found - install with: pip install peft")
    
    try:
        import bitsandbytes
        print(f"  ✓ BitsAndBytes available")
    except ImportError:
        print("  ✗ BitsAndBytes not found - install with: pip install bitsandbytes")
