import os
from pathlib import Path

data_dir = Path('data')

# Check structure
train_dir = data_dir / 'Train'
test_dir = data_dir / 'Test'

print("Checking dataset structure...")
print(f"Train directory exists: {train_dir.exists()}")
print(f"Test directory exists: {test_dir.exists()}")

if train_dir.exists():
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"\nNumber of classes: {len(classes)}")
    print(f"Classes: {classes[:5]}... (showing first 5)")
    
    # Count images in first class
    first_class = train_dir / classes[0]
    images = list(first_class.glob('*.png')) + list(first_class.glob('*.ppm'))
    print(f"\nImages in class '{classes[0]}': {len(images)}")
    
print("\n✅ Dataset verification complete!")