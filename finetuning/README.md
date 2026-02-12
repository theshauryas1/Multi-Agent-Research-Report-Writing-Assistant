# Fine-Tuning Pipeline for Reviewer & Fact-Checker Agents

This directory contains the complete fine-tuning pipeline for training two specialized agents using Phi-2 with QLoRA, optimized for GTX 1650 (4GB VRAM).

## 🎯 Overview

**Two Fine-Tuned Models:**
1. **Reviewer Agent** - Quality classifier for research paper sections
   - Dataset: OpenAssistant (rated responses)
   - Output: `good` / `acceptable` / `weak`
   - Metrics: Accuracy, Precision, Recall, F1

2. **Fact-Checker Agent** - Hallucination detector
   - Dataset: FEVER (fact verification)
   - Output: `SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO`
   - Metrics: Accuracy, Recall on REFUTES

**Base Model:** Phi-2 (2.7B) with 4-bit QLoRA

## 📁 Directory Structure

```
finetuning/
├── config.py                    # Central configuration
├── requirements_finetuning.txt  # Dependencies
├── datasets/
│   ├── prepare_reviewer_dataset.py
│   ├── prepare_factchecker_dataset.py
│   ├── reviewer_data/          # Generated datasets
│   └── factchecker_data/       # Generated datasets
├── utils/
│   ├── gpu_check.py            # GPU verification
│   └── model_loader.py         # Model loading utilities
├── models/
│   ├── phi2-reviewer-lora/     # Reviewer LoRA adapter
│   └── phi2-factchecker-lora/  # Fact-checker LoRA adapter
├── train_reviewer.py           # Training script for reviewer
├── train_factchecker.py        # Training script for fact-checker
├── evaluate_models.py          # Evaluation pipeline
├── results/                    # Evaluation results
└── logs/                       # Training logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r finetuning/requirements_finetuning.txt
```

### 2. Verify GPU

```bash
python finetuning/utils/gpu_check.py
```

Expected output: GTX 1650 detected with ~4GB VRAM

### 3. Prepare Datasets

```bash
# Prepare Reviewer dataset (OpenAssistant)
python finetuning/datasets/prepare_reviewer_dataset.py

# Prepare Fact-Checker dataset (FEVER)
python finetuning/datasets/prepare_factchecker_dataset.py
```

This will download and process 5k samples per dataset (4k train, 500 val, 500 test).

### 4. Train Models

```bash
# Train Reviewer Agent (~2 hours)
python finetuning/train_reviewer.py

# Train Fact-Checker Agent (~2 hours)
python finetuning/train_factchecker.py
```

### 5. Evaluate Models

```bash
python finetuning/evaluate_models.py
```

Results saved to `finetuning/results/`

## 📊 Expected Performance

### Reviewer Agent
| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|-------------------|-------------------|
| Accuracy | 0.62-0.68 | 0.78-0.84 |
| F1 | 0.60-0.65 | 0.76-0.83 |

### Fact-Checker Agent
| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|-------------------|-------------------|
| Accuracy | 0.58-0.65 | 0.72-0.80 |
| Recall (REFUTES) | 0.45-0.55 | 0.68-0.78 |

**Note:** These are realistic, defensible metrics for Phi-2 QLoRA on GTX 1650.

## 🔧 Configuration

All configuration is centralized in `config.py`:

- **Model Config**: Base model, quantization settings
- **LoRA Config**: r=8, alpha=16, dropout=0.05
- **Training Config**: Batch size=1, grad accumulation=16, lr=2e-4, epochs=3
- **Dataset Config**: Dataset sizes, label mappings
- **Path Config**: Model and data paths

## 💡 Usage in LangGraph

### Reviewer Agent

```python
from agents.reviewer_agent import ReviewerAgent

# Initialize
reviewer = ReviewerAgent()

# Review text
verdict = reviewer.review("The model was trained on data...")
# Output: 'weak', 'acceptable', or 'good'

# Get quality score
score = reviewer.get_quality_score("Well-documented methodology...")
# Output: 0.0-1.0
```

### Fact-Checker Agent

```python
from agents.factchecker_agent import FactCheckerAgent

# Initialize
factchecker = FactCheckerAgent()

# Verify claim
status = factchecker.verify_claim(
    claim="CNN achieved 99% accuracy",
    evidence="No benchmark cited"
)
# Output: 'SUPPORTS', 'REFUTES', or 'NOT_ENOUGH_INFO'

# Check for hallucination
is_halluc = factchecker.is_hallucination(claim, evidence)
# Output: True if REFUTES
```

### Paper Formatter

```python
from utils.paper_formatter import PaperFormatter

formatter = PaperFormatter()

# Validate structure
validation = formatter.validate_structure(sections)
# Returns: {'valid': bool, 'errors': [...], 'warnings': [...]}

# Reorder sections
ordered_sections = formatter.reorder_sections(sections)
```

## 🧪 Testing

Each module includes test code:

```bash
# Test GPU
python finetuning/utils/gpu_check.py

# Test model loading
python finetuning/utils/model_loader.py

# Test Reviewer Agent
python agents/reviewer_agent.py

# Test Fact-Checker Agent
python agents/factchecker_agent.py

# Test Paper Formatter
python utils/paper_formatter.py
```

## 📝 Training Details

### QLoRA Configuration
- **Quantization**: 4-bit NF4
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, k_proj, v_proj, dense

### Training Hyperparameters
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 16 steps
- **Effective Batch Size**: 16
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Optimizer**: Paged AdamW 8-bit
- **Scheduler**: Cosine
- **Max Length**: 256 tokens

### Memory Optimization
- 4-bit quantization
- Gradient checkpointing
- Paged optimizer
- Small batch size with gradient accumulation

## 🔍 Troubleshooting

### Out of Memory (OOM)
- Reduce `train_size` in `config.py`
- Ensure no other GPU processes running
- Reduce `max_length` if needed

### Dataset Download Issues
- Check internet connection
- Datasets will be cached in `~/.cache/huggingface/`
- Manually download if needed

### Low Performance
- Ensure GPU is being used (check with `gpu_check.py`)
- Verify dataset quality and balance
- Check training logs for convergence

## 📚 References

- **Phi-2**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- **QLoRA**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- **PEFT**: [HuggingFace PEFT](https://github.com/huggingface/peft)
- **OpenAssistant**: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- **FEVER**: [FEVER Dataset](https://fever.ai/)

## 📄 License

This fine-tuning pipeline is part of the Multi-Agent Research Assistant project.
