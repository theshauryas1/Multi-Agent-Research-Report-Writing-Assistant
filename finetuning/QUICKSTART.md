# Quick Start Guide - Fine-Tuning Pipeline

## 🚀 5-Step Workflow

### Step 1: Install Dependencies (5 minutes)
```bash
cd "d:\research multi level agent"
pip install -r finetuning/requirements_finetuning.txt
```

### Step 2: Verify GPU (1 minute)
```bash
python finetuning/utils/gpu_check.py
```
Expected: GTX 1650 detected with ~4GB VRAM

### Step 3: Prepare Datasets (10 minutes)
```bash
# Reviewer dataset (OpenAssistant)
python finetuning/datasets/prepare_reviewer_dataset.py

# Fact-checker dataset (FEVER)
python finetuning/datasets/prepare_factchecker_dataset.py
```
Creates 5k samples per dataset (4k train, 500 val, 500 test)

### Step 4: Train Models (4 hours total)
```bash
# Train Reviewer Agent (~2 hours)
python finetuning/train_reviewer.py

# Train Fact-Checker Agent (~2 hours)
python finetuning/train_factchecker.py
```

### Step 5: Evaluate & Test (10 minutes)
```bash
# Evaluate both models
python finetuning/evaluate_models.py

# Test integration
python test_integration.py
```

## 📊 Expected Results

**Reviewer Agent:**
- Accuracy: 0.78-0.84 (vs 0.62-0.68 base)
- F1: 0.76-0.83 (vs 0.60-0.65 base)

**Fact-Checker Agent:**
- Accuracy: 0.72-0.80 (vs 0.58-0.65 base)
- Recall (REFUTES): 0.68-0.78 (vs 0.45-0.55 base)

## 💡 Usage

```python
# Fact-checking
from agents.factchecker_agent import get_factchecker_agent
factchecker = get_factchecker_agent()
status = factchecker.verify_claim(claim, evidence)

# Paper formatting
from utils.paper_formatter import get_paper_formatter
formatter = get_paper_formatter()
validation = formatter.validate_structure(sections)
```

## 📚 Documentation

- Full README: `finetuning/README.md`
- Walkthrough: See artifacts
- Integration examples: `test_integration.py`

## ⚠️ Important

- Close other GPU applications before training
- Training takes ~4 hours total (can run overnight)
- Monitor with `nvidia-smi` during training
