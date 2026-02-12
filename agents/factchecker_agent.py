"""
Fact-Checker Agent
Hallucination detection agent for verifying claims against evidence
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetuning.config import PATH_CONFIG, FACTCHECKER_INSTRUCTION, get_prompt_template
from finetuning.utils.model_loader import load_model_with_adapter, generate_text


class FactCheckerAgent:
    """
    Fact-Checker Agent for hallucination detection
    Verifies claims against evidence and flags potential hallucinations
    """
    
    def __init__(self, adapter_path=None):
        """
        Initialize Fact-Checker Agent
        
        Args:
            adapter_path: Path to LoRA adapter (default: from config)
        """
        if adapter_path is None:
            adapter_path = PATH_CONFIG.factchecker_adapter_path
        
        print(f"Loading Fact-Checker Agent from {adapter_path}...")
        self.model, self.tokenizer = load_model_with_adapter(adapter_path)
        self.valid_labels = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        print("✅ Fact-Checker Agent loaded")
    
    def verify_claim(self, claim, evidence, return_confidence=False):
        """
        Verify claim against evidence
        
        Args:
            claim: Claim to verify
            evidence: Evidence to check against
            return_confidence: Whether to return confidence score
        
        Returns:
            str or tuple: Verification status (and confidence if requested)
        """
        # Create input
        input_text = f"Claim: {claim}\nEvidence: {evidence}"
        
        # Create prompt
        prompt = get_prompt_template(FACTCHECKER_INSTRUCTION, input_text)
        
        # Generate response
        response = generate_text(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_new_tokens=15, 
            temperature=0.1
        )
        
        # Extract label
        status = self._extract_label(response)
        
        if return_confidence:
            # Simple confidence based on response clarity
            confidence = 0.9 if status in response else 0.6
            return status, confidence
        
        return status
    
    def _extract_label(self, response):
        """
        Extract verification label from response
        
        Args:
            response: Model response
        
        Returns:
            str: Extracted label
        """
        response_upper = response.upper().strip()
        
        # Check for each label
        for label in self.valid_labels:
            if label in response_upper:
                return label
        
        # Default to NOT_ENOUGH_INFO if unclear
        return 'NOT_ENOUGH_INFO'
    
    def batch_verify(self, claim_evidence_pairs):
        """
        Verify multiple claim-evidence pairs
        
        Args:
            claim_evidence_pairs: List of (claim, evidence) tuples
        
        Returns:
            list: List of verification statuses
        """
        return [self.verify_claim(claim, evidence) for claim, evidence in claim_evidence_pairs]
    
    def is_hallucination(self, claim, evidence):
        """
        Check if claim is a hallucination (refuted by evidence)
        
        Args:
            claim: Claim to check
            evidence: Evidence to check against
        
        Returns:
            bool: True if claim is refuted (hallucination)
        """
        status = self.verify_claim(claim, evidence)
        return status == 'REFUTES'
    
    def get_verification_score(self, claim, evidence):
        """
        Get numerical verification score
        
        Args:
            claim: Claim to verify
            evidence: Evidence to check against
        
        Returns:
            float: Verification score (0-1, higher = more supported)
        """
        status = self.verify_claim(claim, evidence)
        
        score_map = {
            'REFUTES': 0.0,
            'NOT_ENOUGH_INFO': 0.5,
            'SUPPORTS': 1.0
        }
        
        return score_map.get(status, 0.5)


# Singleton instance
_factchecker_agent = None


def get_factchecker_agent():
    """
    Get singleton Fact-Checker Agent instance
    
    Returns:
        FactCheckerAgent: Fact-checker agent instance
    """
    global _factchecker_agent
    
    if _factchecker_agent is None:
        _factchecker_agent = FactCheckerAgent()
    
    return _factchecker_agent


if __name__ == "__main__":
    # Test the agent
    agent = FactCheckerAgent()
    
    test_cases = [
        ("CNN achieved 99% accuracy on unseen MRI data.", "No cited benchmark or validation set mentioned."),
        ("The model was trained on ImageNet dataset.", "We used ImageNet-1K with 1.2M training images."),
        ("Our approach outperforms all baselines.", "Results show 92% accuracy vs 88% for best baseline.")
    ]
    
    print("\n" + "=" * 60)
    print("Testing Fact-Checker Agent")
    print("=" * 60)
    
    for i, (claim, evidence) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Claim: {claim}")
        print(f"Evidence: {evidence}")
        status, confidence = agent.verify_claim(claim, evidence, return_confidence=True)
        score = agent.get_verification_score(claim, evidence)
        is_halluc = agent.is_hallucination(claim, evidence)
        print(f"Status: {status} (confidence: {confidence:.2f}, score: {score:.2f})")
        print(f"Hallucination: {is_halluc}")
