"""
Prepare Reviewer Agent Dataset - Simplified Version
Creates synthetic quality-labeled dataset for training
"""

import os
import json
import sys
import random
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_CONFIG, REVIEWER_INSTRUCTION, get_prompt_template


def create_synthetic_dataset():
    """
    Create synthetic dataset with quality labels
    Uses text patterns to generate realistic examples
    """
    print("=" * 60)
    print("Preparing Reviewer Agent Dataset (Synthetic)")
    print("=" * 60)
    
    print("\n[*] Generating synthetic quality-labeled samples...")
    
    # Templates for different quality levels
    weak_templates = [
        "The model was trained on data.",
        "We used deep learning.",
        "Results were good.",
        "The system works well.",
        "This is important research.",
        "Machine learning is useful.",
        "The algorithm performed adequately.",
        "Data was collected and analyzed.",
        "The experiment showed results.",
        "This approach has potential."
    ]
    
    acceptable_templates = [
        "We trained a convolutional neural network on ImageNet dataset achieving 85% accuracy.",
        "The model uses attention mechanisms to improve performance on sequence tasks.",
        "Our approach combines transfer learning with fine-tuning on domain-specific data.",
        "The system was evaluated on three benchmark datasets with consistent results.",
        "We implemented a novel architecture based on transformer blocks.",
        "The methodology follows standard practices for reproducibility.",
        "Results demonstrate improvement over baseline methods by 5-10%.",
        "The dataset contains 10,000 labeled samples split into train/val/test sets.",
        "We used cross-validation to ensure robust performance estimates.",
        "The approach leverages pre-trained embeddings for better generalization."
    ]
    
    good_templates = [
        "We propose a novel attention-based architecture that achieves state-of-the-art results on GLUE benchmark (92.3% average score), outperforming BERT-large by 2.1 points. The model uses 40% fewer parameters through efficient factorization.",
        "Our method combines self-supervised pre-training on 100M unlabeled samples with supervised fine-tuning on task-specific data. Extensive ablation studies (Table 2) demonstrate that each component contributes significantly to final performance.",
        "The proposed framework integrates three key innovations: (1) dynamic attention weighting, (2) multi-scale feature fusion, and (3) adversarial training. Experiments on five benchmark datasets show consistent improvements of 8-15% over previous best methods.",
        "We conducted rigorous evaluation using 5-fold cross-validation on three datasets (MNIST, CIFAR-10, ImageNet-1K). Results show our approach achieves 95.2±0.3% accuracy on ImageNet, with detailed error analysis provided in Section 4.3.",
        "The architecture employs residual connections and layer normalization, trained end-to-end using Adam optimizer (lr=1e-4, batch=32) for 100 epochs. Hyperparameters were tuned via grid search on validation set, with full details in Appendix A.",
        "Our contribution is three-fold: (1) a novel loss function that balances precision and recall, (2) an efficient inference algorithm reducing latency by 50%, and (3) comprehensive benchmarking against 12 baseline methods across 6 datasets.",
        "The model was trained on 1.2M samples from diverse sources, with careful attention to class balance and data augmentation. We report precision (0.94), recall (0.91), and F1 (0.92) on held-out test set, with confidence intervals computed via bootstrap.",
        "Ablation studies reveal that removing the attention mechanism decreases performance by 12%, while removing residual connections causes a 8% drop. These findings validate our architectural choices (see Table 3 for full results).",
        "We compare against 8 state-of-the-art methods using identical experimental protocols. Statistical significance testing (paired t-test, p<0.01) confirms our improvements are not due to random variation.",
        "The methodology is fully reproducible: code, trained models, and datasets are publicly available at github.com/example. Training takes 24 hours on 4x V100 GPUs with detailed hardware specifications provided."
    ]
    
    samples = []
    
    # Generate samples for each quality level
    for template in weak_templates:
        for _ in range(200):  # 2000 weak samples
            text = template + " " + random.choice([
                "More research is needed.",
                "This is an interesting area.",
                "The results are promising.",
                "Further work will explore this.",
                "This has many applications."
            ])
            samples.append({'text': text, 'label': 'weak'})
    
    for template in acceptable_templates:
        for _ in range(200):  # 2000 acceptable samples
            text = template + " " + random.choice([
                "The implementation uses PyTorch framework.",
                "Training took approximately 12 hours on GPU.",
                "We used standard evaluation metrics.",
                "The code will be made available upon publication.",
                "Experiments were conducted on standard hardware."
            ])
            samples.append({'text': text, 'label': 'acceptable'})
    
    for template in good_templates:
        for _ in range(200):  # 2000 good samples
            text = template + " " + random.choice([
                "Detailed experimental protocols are documented in supplementary materials.",
                "Statistical analysis confirms robustness across different random seeds.",
                "The approach generalizes well to out-of-distribution test cases.",
                "Computational complexity is O(n log n), making it scalable to large datasets.",
                "Limitations and failure cases are discussed in Section 5."
            ])
            samples.append({'text': text, 'label': 'good'})
    
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
        formatted_sample = {
            'instruction': REVIEWER_INSTRUCTION,
            'input': sample['text'],
            'output': sample['label'],
            'prompt': get_prompt_template(
                REVIEWER_INSTRUCTION,
                sample['text'],
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
        'dataset_name': 'synthetic_quality_dataset',
        'total_samples': sum(len(d) for d in datasets.values()),
        'splits': {name: len(data) for name, data in datasets.items()},
        'labels': ['weak', 'acceptable', 'good'],
        'instruction': REVIEWER_INSTRUCTION
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   [OK] Saved metadata")
    
    print("\n" + "=" * 60)
    print("[OK] Reviewer dataset preparation complete!")
    print("=" * 60)


def main():
    """Main execution"""
    # Create synthetic dataset
    samples = create_synthetic_dataset()
    
    # Split dataset
    datasets = split_dataset(samples)
    
    # Save datasets
    save_datasets(datasets, DATASET_CONFIG.reviewer_output_dir)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
