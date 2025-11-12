print("Script started")
import torch
print("PyTorch loaded")
from pathlib import Path
data_dir = Path('data')
happy_images = list((data_dir / 'happy').glob('*.jpg'))
print(f"Found {len(happy_images)} happy images")
sad_images = list((data_dir / 'sad').glob('*.jpg'))
print(f"Found {len(sad_images)} sad images")