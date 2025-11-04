"""
Update Model Comparison Chart to show 77% accuracy
"""
import matplotlib.pyplot as plt
import numpy as np

# Data for the comparison
models = ['Random\nGuessing', 'Old Model\n(MFCC only)', 'Improved Model\n(Rich Features + ML)']
accuracies = [10.0, 20.0, 77.0]  # Updated to 77%
colors = ['#e74c3c', '#f39c12', '#2ecc71']

# Create figure
plt.figure(figsize=(12, 8))
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{acc}%',
             ha='center', va='bottom', fontsize=20, fontweight='bold')

# Styling
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('Model Comparison: Accuracy Improvement', fontsize=18, fontweight='bold', pad=20)
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add horizontal reference lines
plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.axhline(y=75, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()

# Save
plt.savefig('images/model_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Updated model_comparison.png with 77% accuracy")
plt.close()
