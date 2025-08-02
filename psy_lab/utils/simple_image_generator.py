"""
Simple text-based image generator for testing vision experiments.
This is a fallback when matplotlib is not available.
"""

import random
from typing import Tuple, List

def generate_simple_asch_lines(seed: int = None) -> Tuple[str, str]:
    """
    Generate a simple text-based line comparison for the Asch experiment.
    
    Returns:
        Tuple of (text_description, correct_answer)
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate line lengths
    target_length = random.uniform(3, 8)
    correct_length = target_length
    wrong_length = target_length * random.uniform(0.7, 0.9)
    
    # Create text representation
    lines = []
    correct_line_idx = random.randint(0, 2)
    
    for i in range(3):
        if i == correct_line_idx:
            length = correct_length
            marker = "✓"
        else:
            length = wrong_length
            marker = "✗"
        
        line_label = chr(65 + i)  # A, B, C
        lines.append(f"Line {line_label}: {'=' * int(length * 2)} ({length:.1f}) {marker}")
    
    # Create the text "image"
    image_text = f"""
TARGET LINE: {'=' * int(target_length * 2)} (Length: {target_length:.1f})

COMPARISON LINES:
{lines[0]}
{lines[1]}
{lines[2]}

Which line (A, B, or C) matches the target line in length?
"""
    
    correct_answer = chr(65 + correct_line_idx)
    
    return image_text, correct_answer

def generate_multiple_simple_trials(num_trials: int = 5, base_seed: int = 42) -> List[Tuple[str, str]]:
    """
    Generate multiple simple Asch line comparison trials.
    
    Args:
        num_trials: Number of trials to generate
        base_seed: Base seed for reproducibility
        
    Returns:
        List of (text_description, correct_answer) tuples
    """
    trials = []
    for i in range(num_trials):
        seed = base_seed + i
        text, answer = generate_simple_asch_lines(seed)
        trials.append((text, answer))
    return trials

if __name__ == "__main__":
    # Test the simple image generation
    text, answer = generate_simple_asch_lines(42)
    print(f"Generated text image with correct answer: {answer}")
    print("\nText representation:")
    print(text) 