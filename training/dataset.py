"""
Dataset Loader - Prepares training data for fine-tuning.
Formats instruction-response pairs for academic writing style.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainingSample:
    """A single training sample."""
    instruction: str
    input_text: str
    output: str
    source: str = ""
    

class DatasetLoader:
    """
    Loads and processes training datasets.
    
    Supports various formats:
    - JSON Lines (instruction/input/output)
    - Alpaca format
    - Custom research paper format
    """
    
    def __init__(
        self,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the dataset loader.
        
        Args:
            max_samples: Maximum samples to load
            shuffle: Shuffle the dataset
            seed: Random seed for reproducibility
        """
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
    
    def load_jsonl(self, file_path: str) -> List[TrainingSample]:
        """Load dataset from JSON Lines file."""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                sample = TrainingSample(
                    instruction=data.get('instruction', ''),
                    input_text=data.get('input', ''),
                    output=data.get('output', ''),
                    source=data.get('source', file_path),
                )
                samples.append(sample)
        
        return self._process_samples(samples)
    
    def load_json(self, file_path: str) -> List[TrainingSample]:
        """Load dataset from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            samples = [
                TrainingSample(
                    instruction=item.get('instruction', ''),
                    input_text=item.get('input', ''),
                    output=item.get('output', ''),
                    source=item.get('source', file_path),
                )
                for item in data
            ]
        else:
            raise ValueError("JSON file should contain a list of samples")
        
        return self._process_samples(samples)
    
    def load_directory(
        self, 
        directory: str,
        extensions: List[str] = ['.jsonl', '.json'],
    ) -> List[TrainingSample]:
        """Load all dataset files from a directory."""
        samples = []
        dir_path = Path(directory)
        
        for ext in extensions:
            for file_path in dir_path.glob(f'*{ext}'):
                try:
                    if ext == '.jsonl':
                        file_samples = self.load_jsonl(str(file_path))
                    else:
                        file_samples = self.load_json(str(file_path))
                    samples.extend(file_samples)
                    print(f"✓ Loaded {len(file_samples)} samples from {file_path.name}")
                except Exception as e:
                    print(f"✗ Failed to load {file_path.name}: {e}")
        
        return self._process_samples(samples)
    
    def _process_samples(self, samples: List[TrainingSample]) -> List[TrainingSample]:
        """Process and filter samples."""
        # Filter empty samples
        samples = [s for s in samples if s.output.strip()]
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(samples)
        
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        return samples
    
    def format_for_training(
        self,
        samples: List[TrainingSample],
        template: str = "mistral",
    ) -> List[Dict[str, str]]:
        """
        Format samples for training with chat template.
        
        Args:
            samples: List of training samples
            template: Template format ('mistral', 'alpaca', 'chatml')
            
        Returns:
            List of formatted training examples
        """
        formatted = []
        
        for sample in samples:
            if template == "mistral":
                text = self._format_mistral(sample)
            elif template == "alpaca":
                text = self._format_alpaca(sample)
            elif template == "chatml":
                text = self._format_chatml(sample)
            else:
                text = self._format_simple(sample)
            
            formatted.append({"text": text})
        
        return formatted
    
    def _format_mistral(self, sample: TrainingSample) -> str:
        """Format for Mistral instruction format."""
        if sample.input_text:
            prompt = f"{sample.instruction}\n\n{sample.input_text}"
        else:
            prompt = sample.instruction
        
        return f"<s>[INST] {prompt} [/INST] {sample.output}</s>"
    
    def _format_alpaca(self, sample: TrainingSample) -> str:
        """Format for Alpaca template."""
        if sample.input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample.instruction}

### Input:
{sample.input_text}

### Response:
{sample.output}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample.instruction}

### Response:
{sample.output}"""
    
    def _format_chatml(self, sample: TrainingSample) -> str:
        """Format for ChatML template."""
        if sample.input_text:
            user_content = f"{sample.instruction}\n\n{sample.input_text}"
        else:
            user_content = sample.instruction
        
        return f"""<|im_start|>system
You are a helpful research assistant that writes clear, academically-styled content.<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{sample.output}<|im_end|>"""
    
    def _format_simple(self, sample: TrainingSample) -> str:
        """Simple format."""
        if sample.input_text:
            return f"Instruction: {sample.instruction}\nInput: {sample.input_text}\nOutput: {sample.output}"
        return f"Instruction: {sample.instruction}\nOutput: {sample.output}"
    
    def split_dataset(
        self,
        samples: List[Any],
        train_ratio: float = 0.9,
    ) -> Tuple[List[Any], List[Any]]:
        """Split dataset into train and validation sets."""
        split_idx = int(len(samples) * train_ratio)
        return samples[:split_idx], samples[split_idx:]


def create_training_dataset(
    data_path: str,
    template: str = "mistral",
    train_ratio: float = 0.9,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create training and validation datasets.
    
    Args:
        data_path: Path to data file or directory
        template: Formatting template
        train_ratio: Train/val split ratio
        max_samples: Maximum samples to use
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    loader = DatasetLoader(max_samples=max_samples)
    
    path = Path(data_path)
    if path.is_dir():
        samples = loader.load_directory(str(path))
    elif path.suffix == '.jsonl':
        samples = loader.load_jsonl(str(path))
    else:
        samples = loader.load_json(str(path))
    
    print(f"Loaded {len(samples)} samples")
    
    # Format for training
    formatted = loader.format_for_training(samples, template=template)
    
    # Split
    train_data, val_data = loader.split_dataset(formatted, train_ratio)
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    return train_data, val_data


def create_research_samples() -> List[TrainingSample]:
    """
    Create sample training data for research writing.
    
    These are examples - in practice, you'd load from actual papers.
    """
    samples = [
        TrainingSample(
            instruction="Write an introduction section for a research paper on renewable energy.",
            input_text="Focus on solar and wind power, their growth trends, and importance for climate change mitigation.",
            output="""Renewable energy sources have emerged as critical components in the global transition toward sustainable power generation. Over the past decade, solar photovoltaic and wind power installations have grown at unprecedented rates, with combined capacity increasing by approximately 25% annually. This rapid expansion is driven by declining costs, supportive policies, and growing recognition of climate change imperatives.

Solar power, in particular, has seen remarkable cost reductions, with levelized costs falling by over 80% since 2010. Similarly, onshore wind power has achieved cost parity with fossil fuel generation in many regions. These economic factors, combined with technological improvements in efficiency and storage, have accelerated adoption across both developed and developing nations.

The transition to renewable energy is essential for achieving the emissions reductions targets outlined in the Paris Agreement. This paper examines current trends in solar and wind power deployment, analyzes the technical and economic factors driving growth, and discusses the policy frameworks that have proven most effective in accelerating the clean energy transition.""",
            source="synthetic_research",
        ),
        TrainingSample(
            instruction="Write a methodology section describing a multi-agent system for report generation.",
            input_text="The system uses LangGraph for orchestration, includes research, planning, writing, and review agents.",
            output="""This study employs a multi-agent architecture for automated report generation, implemented using the LangGraph framework for workflow orchestration. The system comprises five specialized agents, each designed to handle distinct aspects of the report generation pipeline.

The Research Agent conducts information retrieval using web search APIs and maintains a vector database for relevant document chunks. Retrieved information is processed through a semantic chunking pipeline to preserve contextual coherence. The Planner Agent receives aggregated research findings and generates a structured outline that defines the logical flow and key topics for each section.

The Writer Agent generates content section-by-section, utilizing the outline and research context to produce coherent prose. Each section is passed to the Reviewer Agent, which evaluates content quality across multiple dimensions: clarity, accuracy, structure, and source grounding. Sections failing to meet quality thresholds are routed to the Fixer Agent for revision.

The workflow follows a directed acyclic graph pattern with conditional revision loops. Maximum revision iterations are capped at three to prevent infinite loops while allowing sufficient opportunity for quality improvement. State management ensures consistent context propagation across agent interactions.""",
            source="synthetic_research",
        ),
        TrainingSample(
            instruction="Summarize the key findings of a study on AI-assisted writing.",
            input_text="Focus on improvements in productivity, quality metrics, and user satisfaction.",
            output="""The evaluation of the AI-assisted writing system revealed significant improvements across multiple dimensions. Productivity metrics showed a 3.2x reduction in total document creation time, with the most substantial gains observed in initial research and outline generation phases. Writers using the system completed first drafts 65% faster than the control group working without AI assistance.

Quality assessments, conducted using a standardized rubric scored by domain experts, indicated consistent improvements in structural organization and citation accuracy. AI-assisted documents received an average quality score of 8.4/10 compared to 7.1/10 for manually produced documents. Notably, error rates for factual claims decreased from 8.2% to 2.1% when the AI system's source grounding was enabled.

User satisfaction surveys revealed high acceptance rates, with 87% of participants indicating they would continue using the system for professional writing tasks. Participants particularly valued the system's ability to maintain consistency across long documents and to suggest relevant citations from the retrieved source material.""",
            source="synthetic_research",
        ),
    ]
    
    return samples


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loader...")
    
    # Create sample data
    samples = create_research_samples()
    print(f"Created {len(samples)} sample training examples")
    
    # Format for training
    loader = DatasetLoader()
    formatted = loader.format_for_training(samples, template="mistral")
    
    print(f"\nSample formatted output:")
    print("-" * 50)
    print(formatted[0]["text"][:500] + "...")
