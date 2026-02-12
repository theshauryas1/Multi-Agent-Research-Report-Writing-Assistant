"""
Model Loading Utilities
Centralized functions for loading Phi-2 base model and LoRA adapters
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, LORA_CONFIG


def load_base_model(model_name=None):
    """
    Load Phi-2 base model with 4-bit quantization
    
    Args:
        model_name: Model name/path (default: from config)
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_name is None:
        model_name = MODEL_CONFIG.base_model_name
    
    print(f"Loading base model: {model_name}")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=MODEL_CONFIG.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=MODEL_CONFIG.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=MODEL_CONFIG.bnb_4bit_use_double_quant
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=MODEL_CONFIG.device_map,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("✅ Base model loaded successfully")
    
    return model, tokenizer


def prepare_model_for_training(model):
    """
    Prepare model for QLoRA training
    
    Args:
        model: Base model
    
    Returns:
        model: Prepared model
    """
    print("Preparing model for training...")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("✅ Model prepared for training")
    
    return model


def load_model_with_adapter(adapter_path, model_name=None):
    """
    Load base model with LoRA adapter
    
    Args:
        adapter_path: Path to LoRA adapter
        model_name: Base model name (default: from config)
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model with adapter: {adapter_path}")
    
    # Load base model
    base_model, tokenizer = load_base_model(model_name)
    
    # Load adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False
    )
    
    print("✅ Model with adapter loaded successfully")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    """
    Generate text using the model
    
    Args:
        model: Model (base or with adapter)
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        str: Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MODEL_CONFIG.max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove prompt)
    generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def get_model_memory_usage(model):
    """
    Get model memory usage
    
    Args:
        model: Model
    
    Returns:
        dict: Memory usage statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated
    }


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    
    try:
        model, tokenizer = load_base_model()
        
        # Print memory usage
        memory = get_model_memory_usage(model)
        print(f"\nMemory Usage:")
        print(f"  Allocated: {memory['allocated_gb']:.2f} GB")
        print(f"  Reserved: {memory['reserved_gb']:.2f} GB")
        
        # Test generation
        test_prompt = "### Instruction:\nExplain quantum computing.\n\n### Response:\n"
        print(f"\nTest generation:")
        print(f"Prompt: {test_prompt}")
        
        output = generate_text(model, tokenizer, test_prompt, max_new_tokens=30)
        print(f"Output: {output}")
        
        print("\n✅ Model loading test successful")
        
    except Exception as e:
        print(f"\n❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
