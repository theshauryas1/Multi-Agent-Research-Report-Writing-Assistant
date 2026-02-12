"""
Prepare Fact-Checker Agent Dataset - Simplified Version
Creates synthetic claim-evidence dataset for training
"""

import os
import json
import sys
import random
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_CONFIG, FACTCHECKER_INSTRUCTION, get_prompt_template


def create_synthetic_dataset():
    """
    Create synthetic fact-checking dataset
    """
    print("=" * 60)
    print("Preparing Fact-Checker Agent Dataset (Synthetic)")
    print("=" * 60)
    
    print("\n[*] Generating synthetic claim-evidence pairs...")
    
    samples = []
    
    # SUPPORTS examples
    supports_pairs = [
        ("The model achieves 95% accuracy on the test set.", "Table 1 shows test accuracy of 95.2% across 5 runs."),
        ("Training used the Adam optimizer.", "We trained the model using Adam optimizer with learning rate 1e-4."),
        ("The dataset contains 10,000 samples.", "We collected 10,000 labeled samples for our experiments."),
        ("The approach outperforms the baseline.", "Our method achieves 92% accuracy vs 88% for the baseline."),
        ("The model was trained for 100 epochs.", "Training proceeded for 100 epochs with early stopping."),
        ("We used cross-validation for evaluation.", "5-fold cross-validation was employed to assess performance."),
        ("The architecture uses attention mechanisms.", "The model incorporates multi-head self-attention layers."),
        ("Results show statistical significance.", "Paired t-test confirms significance at p<0.01 level."),
        ("The dataset is publicly available.", "Data and code are released at github.com/example/repo."),
        ("Inference takes 50ms per sample.", "Average inference latency measured at 48±3ms per sample.")
    ]
    
    # REFUTES examples
    refutes_pairs = [
        ("The model achieves 99% accuracy.", "Experimental results show 85% accuracy on the test set."),
        ("This is the first work on this topic.", "Multiple prior works (Smith 2019, Jones 2020) have addressed this problem."),
        ("The approach works on all datasets.", "Our method shows poor performance on Dataset C (45% accuracy)."),
        ("Training takes only 1 hour.", "Model training required 24 hours on 4x V100 GPUs."),
        ("No hyperparameter tuning was needed.", "Extensive grid search was performed to optimize hyperparameters."),
        ("The model has no limitations.", "Section 5 discusses several limitations including sensitivity to noise."),
        ("Results generalize to all domains.", "The approach is specifically designed for medical imaging and may not transfer."),
        ("The dataset is perfectly balanced.", "Class distribution is highly imbalanced (90% negative, 10% positive)."),
        ("No data augmentation was used.", "We applied standard augmentation techniques including rotation and flipping."),
        ("The model is parameter-free.", "The architecture contains 175M trainable parameters.")
    ]
    
    # NOT_ENOUGH_INFO examples
    not_enough_pairs = [
        ("The model uses dropout regularization.", "Training details are not provided in the paper."),
        ("Experiments were run on GPU.", "Hardware specifications are not mentioned."),
        ("The dataset was collected in 2020.", "No information about data collection timeline is given."),
        ("The approach scales to large datasets.", "Scalability analysis is not included in the evaluation."),
        ("The model was pre-trained.", "Pre-training procedures are not documented."),
        ("Batch size was set to 32.", "Training hyperparameters are not specified."),
        ("The code is written in Python.", "Implementation details are not disclosed."),
        ("Results were averaged over 5 runs.", "Number of experimental runs is not reported."),
        ("The model uses batch normalization.", "Architectural details are not fully described."),
        ("Data was split 80/10/10.", "Dataset split ratios are not mentioned.")
    ]
    
    # Generate samples
    for claim, evidence in supports_pairs:
        for _ in range(200):  # 2000 SUPPORTS
            samples.append({
                'claim': claim,
                'evidence': evidence,
                'label': 'SUPPORTS'
            })
    
    for claim, evidence in refutes_pairs:
        for _ in range(200):  # 2000 REFUTES
            samples.append({
                'claim': claim,
                'evidence': evidence,
                'label': 'REFUTES'
            })
    
    for claim, evidence in not_enough_pairs:
        for _ in range(200):  # 2000 NOT_ENOUGH_INFO
            samples.append({
                'claim': claim,
                'evidence': evidence,
                'label': 'NOT_ENOUGH_INFO'
            })
    
    print(f"[OK] Generated {len(samples)} samples")
    
    # Shuffle
    random.shuffle(samples)
    
    # Check distribution
    label_counts = Counter([s['label'] for s in samples])
    print(f"\n[*] Label distribution:")
    for label, count in label_counts.items():
        print(f"   {label}: {count} ({count/len(samples)*100:.1f}%)")
    
    return samples


def split_dataset(samples):
    """Split into train/val/test"""
    total = len(samples)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = samples[:train_size]
    val_data = samples[train_size:train_size + val_size]
    test_data = samples[train_size + val_size:]
    
    print(f"\n[*] Dataset splits:")
    print(f"   Train: {len(train_data)}")
    print(f"   Validation: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    return {
        'train': format_for_training(train_data),
        'validation': format_for_training(val_data),
        'test': format_for_training(test_data)
    }


def format_for_training(samples):
    """Format samples for training"""
    formatted = []
    
    for sample in samples:
        input_text = f"Claim: {sample['claim']}\nEvidence: {sample['evidence']}"
        
        formatted_sample = {
            'instruction': FACTCHECKER_INSTRUCTION,
            'input': input_text,
            'output': sample['label'],
            'prompt': get_prompt_template(
                FACTCHECKER_INSTRUCTION,
                input_text,
                sample['label']
            )
        }
        formatted.append(formatted_sample)
    
    return formatted


def save_datasets(datasets, output_dir):
    """Save datasets to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[*] Saving datasets to {output_dir}...")
    
    for split_name, split_data in datasets.items():
        output_path = os.path.join(output_dir, f"{split_name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"   [OK] Saved {split_name}: {len(split_data)} samples")
    
    # Save metadata
    metadata = {
        'dataset_name': 'synthetic_factcheck_dataset',
        'total_samples': sum(len(d) for d in datasets.values()),
        'splits': {name: len(data) for name, data in datasets.items()},
        'labels': ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'],
        'instruction': FACTCHECKER_INSTRUCTION
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   [OK] Saved metadata")
    
    print("\n" + "=" * 60)
    print("[OK] Fact-checker dataset preparation complete!")
    print("=" * 60)


def main():
    """Main execution"""
    # Create synthetic dataset
    samples = create_synthetic_dataset()
    
    # Split dataset
    datasets = split_dataset(samples)
    
    # Save datasets
    save_datasets(datasets, DATASET_CONFIG.factchecker_output_dir)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
