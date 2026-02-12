"""
Model Evaluation Module
Evaluates fine-tuned models and generates comparison reports
"""

import os
import sys
import json
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from collections import defaultdict
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_CONFIG, PATH_CONFIG, REVIEWER_INSTRUCTION, FACTCHECKER_INSTRUCTION, get_prompt_template
from utils.model_loader import load_base_model, load_model_with_adapter, generate_text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data(dataset_type='reviewer'):
    """
    Load test dataset
    
    Args:
        dataset_type: 'reviewer' or 'factchecker'
    
    Returns:
        list: Test samples
    """
    if dataset_type == 'reviewer':
        data_dir = DATASET_CONFIG.reviewer_output_dir
    else:
        data_dir = DATASET_CONFIG.factchecker_output_dir
    
    test_path = os.path.join(data_dir, 'test.json')
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples for {dataset_type}")
    
    return test_data


def extract_label_from_response(response, valid_labels):
    """
    Extract label from model response
    
    Args:
        response: Model generated response
        valid_labels: List of valid labels
    
    Returns:
        str: Extracted label or None
    """
    response = response.strip().upper()
    
    # Try exact match first
    for label in valid_labels:
        if label.upper() in response:
            return label
    
    # Try partial match
    response_lower = response.lower()
    for label in valid_labels:
        if label.lower() in response_lower:
            return label
    
    return None


def evaluate_model(model, tokenizer, test_data, dataset_type='reviewer'):
    """
    Evaluate model on test data
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_data: Test dataset
        dataset_type: 'reviewer' or 'factchecker'
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating {dataset_type} model...")
    
    if dataset_type == 'reviewer':
        valid_labels = list(DATASET_CONFIG.reviewer_labels.keys())
    else:
        valid_labels = list(DATASET_CONFIG.factchecker_labels.keys())
    
    predictions = []
    ground_truths = []
    
    for i, sample in enumerate(test_data):
        if i % 50 == 0:
            logger.info(f"Processing sample {i}/{len(test_data)}")
        
        # Generate prompt (without output)
        prompt = get_prompt_template(sample['instruction'], sample['input'])
        
        # Generate prediction
        try:
            response = generate_text(model, tokenizer, prompt, max_new_tokens=10, temperature=0.1)
            predicted_label = extract_label_from_response(response, valid_labels)
            
            if predicted_label is None:
                # Default to first label if extraction fails
                predicted_label = valid_labels[0]
            
            predictions.append(predicted_label)
            ground_truths.append(sample['output'])
            
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            predictions.append(valid_labels[0])
            ground_truths.append(sample['output'])
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truths, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_report = classification_report(
        ground_truths, predictions, 
        labels=valid_labels,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_metrics': class_report
    }
    
    logger.info(f"Evaluation complete!")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    
    return metrics


def evaluate_reviewer_agent():
    """
    Evaluate Reviewer Agent (before and after fine-tuning)
    
    Returns:
        dict: Evaluation results
    """
    logger.info("=" * 60)
    logger.info("Evaluating Reviewer Agent")
    logger.info("=" * 60)
    
    # Load test data
    test_data = load_test_data('reviewer')
    
    results = {}
    
    # Evaluate base model
    logger.info("\n📊 Evaluating BASE model...")
    base_model, tokenizer = load_base_model()
    base_metrics = evaluate_model(base_model, tokenizer, test_data, 'reviewer')
    results['base_model'] = base_metrics
    
    # Clear memory
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    logger.info("\n📊 Evaluating FINE-TUNED model...")
    ft_model, tokenizer = load_model_with_adapter(PATH_CONFIG.reviewer_adapter_path)
    ft_metrics = evaluate_model(ft_model, tokenizer, test_data, 'reviewer')
    results['fine_tuned_model'] = ft_metrics
    
    # Clear memory
    del ft_model
    torch.cuda.empty_cache()
    
    return results


def evaluate_factchecker_agent():
    """
    Evaluate Fact-Checker Agent (before and after fine-tuning)
    
    Returns:
        dict: Evaluation results
    """
    logger.info("=" * 60)
    logger.info("Evaluating Fact-Checker Agent")
    logger.info("=" * 60)
    
    # Load test data
    test_data = load_test_data('factchecker')
    
    results = {}
    
    # Evaluate base model
    logger.info("\n📊 Evaluating BASE model...")
    base_model, tokenizer = load_base_model()
    base_metrics = evaluate_model(base_model, tokenizer, test_data, 'factchecker')
    results['base_model'] = base_metrics
    
    # Clear memory
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    logger.info("\n📊 Evaluating FINE-TUNED model...")
    ft_model, tokenizer = load_model_with_adapter(PATH_CONFIG.factchecker_adapter_path)
    ft_metrics = evaluate_model(ft_model, tokenizer, test_data, 'factchecker')
    results['fine_tuned_model'] = ft_metrics
    
    # Clear memory
    del ft_model
    torch.cuda.empty_cache()
    
    return results


def generate_comparison_report(results, agent_type='reviewer'):
    """
    Generate markdown comparison report
    
    Args:
        results: Evaluation results
        agent_type: 'reviewer' or 'factchecker'
    
    Returns:
        str: Markdown report
    """
    report = f"# {agent_type.title()} Agent Evaluation Report\n\n"
    
    report += "## Overall Metrics Comparison\n\n"
    report += "| Metric | Base Model | Fine-Tuned Model | Improvement |\n"
    report += "|--------|------------|------------------|-------------|\n"
    
    base = results['base_model']
    ft = results['fine_tuned_model']
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        base_val = base[metric]
        ft_val = ft[metric]
        improvement = ft_val - base_val
        
        report += f"| {metric.title()} | {base_val:.4f} | {ft_val:.4f} | "
        report += f"{'+' if improvement >= 0 else ''}{improvement:.4f} |\n"
    
    report += "\n## Per-Class Metrics (Fine-Tuned Model)\n\n"
    report += "| Class | Precision | Recall | F1-Score | Support |\n"
    report += "|-------|-----------|--------|----------|----------|\n"
    
    for label, metrics in ft['per_class_metrics'].items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        
        report += f"| {label} | {metrics['precision']:.4f} | "
        report += f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
        report += f"{int(metrics['support'])} |\n"
    
    return report


def save_results(reviewer_results, factchecker_results):
    """
    Save evaluation results
    
    Args:
        reviewer_results: Reviewer evaluation results
        factchecker_results: Fact-checker evaluation results
    """
    os.makedirs(PATH_CONFIG.eval_results_dir, exist_ok=True)
    
    # Save JSON results
    results = {
        'reviewer': reviewer_results,
        'factchecker': factchecker_results
    }
    
    json_path = os.path.join(PATH_CONFIG.eval_results_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Saved JSON results to {json_path}")
    
    # Generate and save markdown reports
    reviewer_report = generate_comparison_report(reviewer_results, 'reviewer')
    factchecker_report = generate_comparison_report(factchecker_results, 'factchecker')
    
    reviewer_md_path = os.path.join(PATH_CONFIG.eval_results_dir, 'reviewer_evaluation.md')
    with open(reviewer_md_path, 'w') as f:
        f.write(reviewer_report)
    
    factchecker_md_path = os.path.join(PATH_CONFIG.eval_results_dir, 'factchecker_evaluation.md')
    with open(factchecker_md_path, 'w') as f:
        f.write(factchecker_report)
    
    logger.info(f"✅ Saved markdown reports to {PATH_CONFIG.eval_results_dir}")


def main():
    """Main evaluation function"""
    logger.info("=" * 60)
    logger.info("Model Evaluation Pipeline")
    logger.info("=" * 60)
    
    # Evaluate Reviewer Agent
    reviewer_results = evaluate_reviewer_agent()
    
    # Evaluate Fact-Checker Agent
    factchecker_results = evaluate_factchecker_agent()
    
    # Save results
    save_results(reviewer_results, factchecker_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Evaluation complete!")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
