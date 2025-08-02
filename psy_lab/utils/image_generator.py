"""
Image generation utilities for the Asch conformity experiment.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import base64
from typing import Tuple, List
import random

def generate_asch_lines(seed: int = None) -> Tuple[str, str]:
    """
    Generate a line comparison image for the Asch experiment.
    
    Returns:
        Tuple of (image_base64, correct_answer)
    """
    if seed is not None:
        random.seed(seed)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Generate line lengths
    target_length = random.uniform(3, 8)
    correct_length = target_length
    wrong_length = target_length * random.uniform(0.7, 0.9)  # Make it noticeably different
    
    # Create target line (left panel)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.set_title('Target Line', fontsize=14, fontweight='bold')
    
    # Draw target line
    target_x = [2, 2 + target_length]
    target_y = [5, 5]
    ax1.plot(target_x, target_y, 'k-', linewidth=4, solid_capstyle='round')
    ax1.text(2 + target_length/2, 6, f'Length: {target_length:.1f}', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Remove axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Create comparison lines (right panel)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.set_title('Which line matches the target?', fontsize=14, fontweight='bold')
    
    # Generate 3 comparison lines
    lines = []
    correct_line_idx = random.randint(0, 2)
    
    for i in range(3):
        if i == correct_line_idx:
            length = correct_length
            color = 'blue'
            label = f'Line {chr(65+i)} (Correct)'
        else:
            length = wrong_length
            color = 'red'
            label = f'Line {chr(65+i)}'
        
        # Position lines vertically
        y_pos = 3 + i * 2
        x_start = 2
        x_end = x_start + length
        
        ax2.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=4, solid_capstyle='round')
        ax2.text(x_end + 0.5, y_pos, f'{chr(65+i)}', fontsize=12, fontweight='bold', va='center')
        ax2.text(5, y_pos, f'Length: {length:.1f}', fontsize=9, ha='center', va='center')
        
        lines.append((chr(65+i), length, i == correct_line_idx))
    
    # Remove axes
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Add legend
    ax2.text(8, 8, 'Legend:', fontsize=10, fontweight='bold')
    ax2.text(8, 7.5, 'Blue = Correct', fontsize=9, color='blue')
    ax2.text(8, 7, 'Red = Incorrect', fontsize=9, color='red')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    correct_answer = chr(65 + correct_line_idx)
    
    return image_base64, correct_answer

def generate_multiple_trials(num_trials: int = 5, base_seed: int = 42) -> List[Tuple[str, str]]:
    """
    Generate multiple Asch line comparison trials.
    
    Args:
        num_trials: Number of trials to generate
        base_seed: Base seed for reproducibility
        
    Returns:
        List of (image_base64, correct_answer) tuples
    """
    trials = []
    for i in range(num_trials):
        seed = base_seed + i
        image, answer = generate_asch_lines(seed)
        trials.append((image, answer))
    return trials

if __name__ == "__main__":
    # Test the image generation
    image, answer = generate_asch_lines(42)
    print(f"Generated image with correct answer: {answer}")
    
    # Save a test image
    import base64
    with open("test_asch_image.png", "wb") as f:
        f.write(base64.b64decode(image))
    print("Saved test image as test_asch_image.png") 