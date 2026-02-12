"""
Integration Example
Demonstrates how to use fine-tuned agents in the research pipeline
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.factchecker_agent import FactCheckerAgent
from utils.paper_formatter import PaperFormatter


def example_factchecker_integration():
    """
    Example: Using Fact-Checker Agent to verify claims
    """
    print("=" * 60)
    print("Fact-Checker Agent Integration Example")
    print("=" * 60)
    
    # Initialize agent
    factchecker = FactCheckerAgent()
    
    # Example claims from a research paper
    claims_and_evidence = [
        {
            "claim": "Our model achieves 95% accuracy on the test set.",
            "evidence": "Table 1 shows test accuracy of 94.8% ± 0.3% across 5 runs.",
            "expected": "SUPPORTS"
        },
        {
            "claim": "This is the first work to apply deep learning to medical imaging.",
            "evidence": "Multiple prior works (Smith 2019, Jones 2020) have used CNNs for medical image analysis.",
            "expected": "REFUTES"
        },
        {
            "claim": "The model was trained for 100 epochs.",
            "evidence": "Training details are not provided in the methodology section.",
            "expected": "NOT_ENOUGH_INFO"
        }
    ]
    
    print("\nVerifying claims...\n")
    
    for i, item in enumerate(claims_and_evidence, 1):
        print(f"Example {i}:")
        print(f"Claim: {item['claim']}")
        print(f"Evidence: {item['evidence']}")
        
        status, confidence = factchecker.verify_claim(
            item['claim'], 
            item['evidence'], 
            return_confidence=True
        )
        
        is_halluc = factchecker.is_hallucination(item['claim'], item['evidence'])
        
        print(f"Verdict: {status} (confidence: {confidence:.2f})")
        print(f"Hallucination: {is_halluc}")
        print(f"Expected: {item['expected']}")
        print()


def example_paper_formatter_integration():
    """
    Example: Using Paper Formatter to validate structure
    """
    print("=" * 60)
    print("Paper Formatter Integration Example")
    print("=" * 60)
    
    # Initialize formatter
    formatter = PaperFormatter()
    
    # Example paper sections (incomplete)
    incomplete_sections = [
        {"title": "Abstract", "content": "This paper presents a novel approach to machine learning."},
        {"title": "Introduction", "content": "Machine learning has become increasingly important..."},
        {"title": "Methodology", "content": "We propose a new neural network architecture..."},
        {"title": "Conclusion", "content": "We have demonstrated the effectiveness of our approach."}
    ]
    
    print("\nValidating incomplete paper structure...\n")
    
    validation = formatter.validate_structure(incomplete_sections)
    
    print(f"Valid: {validation['valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Missing sections: {validation['missing_sections']}")
    
    # Complete paper sections
    complete_sections = [
        {"title": "Abstract", "content": "This paper presents a comprehensive study of deep learning architectures for image classification. We propose a novel CNN-based approach that achieves state-of-the-art results on benchmark datasets."},
        {"title": "Introduction", "content": "Image classification is a fundamental task in computer vision. This work addresses the challenge of improving accuracy while maintaining computational efficiency."},
        {"title": "Related Work", "content": "Prior work in this area includes ResNet (He et al., 2016) and EfficientNet (Tan et al., 2019). Our approach builds upon these foundations."},
        {"title": "Methodology", "content": "We designed a novel architecture combining residual connections with attention mechanisms. The model was trained on ImageNet-1K using SGD optimizer."},
        {"title": "Results", "content": "Our model achieves 95.2% top-1 accuracy on ImageNet validation set, outperforming baseline methods by 2.3%."},
        {"title": "Discussion", "content": "The results demonstrate that our approach effectively balances accuracy and efficiency. Limitations include higher memory requirements."},
        {"title": "Conclusion", "content": "We presented a novel CNN architecture that achieves state-of-the-art results. Future work will explore model compression techniques."},
        {"title": "References", "content": "He et al. (2016). Deep Residual Learning. CVPR.\nTan et al. (2019). EfficientNet. ICML."}
    ]
    
    print("\n" + "=" * 60)
    print("Validating complete paper structure...\n")
    
    validation = formatter.validate_structure(complete_sections)
    
    print(f"Valid: {validation['valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    
    if validation['valid']:
        print("\n✅ Paper structure is valid!")
        
        # Format the paper
        formatted_paper = formatter.format_paper(complete_sections)
        print("\nFormatted paper preview (first 500 chars):")
        print(formatted_paper[:500] + "...")


def example_combined_workflow():
    """
    Example: Combined workflow with both agents
    """
    print("\n" + "=" * 60)
    print("Combined Workflow Example")
    print("=" * 60)
    
    # Initialize agents
    factchecker = FactCheckerAgent()
    formatter = PaperFormatter()
    
    # Simulated research paper sections
    sections = [
        {
            "title": "Abstract",
            "content": "We propose a novel deep learning model that achieves 99.9% accuracy on all benchmark datasets.",
            "claims": [
                ("99.9% accuracy on all benchmarks", "Results show 95.2% on ImageNet, 92.1% on CIFAR-10")
            ]
        },
        {
            "title": "Methodology",
            "content": "Our approach uses a standard CNN architecture trained on ImageNet.",
            "claims": [
                ("Uses standard CNN architecture", "Architecture based on ResNet-50 with modifications")
            ]
        }
    ]
    
    print("\n1. Validating paper structure...")
    validation = formatter.validate_structure(sections)
    print(f"   Structure valid: {validation['valid']}")
    print(f"   Missing sections: {validation['missing_sections']}")
    
    print("\n2. Fact-checking claims...")
    hallucinations_found = []
    
    for section in sections:
        print(f"\n   Section: {section['title']}")
        for claim, evidence in section.get('claims', []):
            status = factchecker.verify_claim(claim, evidence)
            print(f"   - Claim: {claim[:50]}...")
            print(f"     Status: {status}")
            
            if status == 'REFUTES':
                hallucinations_found.append({
                    'section': section['title'],
                    'claim': claim
                })
    
    print(f"\n3. Summary:")
    print(f"   Hallucinations found: {len(hallucinations_found)}")
    print(f"   Paper structure issues: {len(validation['errors'])}")
    
    if hallucinations_found:
        print("\n   ⚠️  Hallucinations detected:")
        for h in hallucinations_found:
            print(f"   - {h['section']}: {h['claim'][:60]}...")
    
    if not validation['valid']:
        print("\n   ⚠️  Structure issues:")
        for error in validation['errors']:
            print(f"   - {error}")


if __name__ == "__main__":
    # Run examples
    example_factchecker_integration()
    print("\n" + "=" * 60 + "\n")
    
    example_paper_formatter_integration()
    print("\n" + "=" * 60 + "\n")
    
    example_combined_workflow()
    
    print("\n" + "=" * 60)
    print("✅ Integration examples complete!")
    print("=" * 60)
