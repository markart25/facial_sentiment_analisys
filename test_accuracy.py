import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

# Emotion labels
EMOTIONS = ['happy', 'sad']
EMOTION_MAP = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# ============ DATASET CLASS ============
class FacialExpressionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        for emotion in EMOTIONS:
            emotion_dir = self.data_dir / emotion
            if emotion_dir.exists():
                for img_file in emotion_dir.glob('*.jpg'):
                    self.images.append(str(img_file))
                    self.labels.append(EMOTION_MAP[emotion])
            else:
                print(f"Warning: {emotion_dir} not found")
        
        print(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ============ CNN MODEL ============
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============ TESTING ============
def test_model(test_data_dir='test_data', model_path='emotion_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load test dataset
    print(f"\nLoading test data from {test_data_dir}...")
    test_dataset = FacialExpressionDataset(test_data_dir, transform=transform)
    if len(test_dataset) == 0:
        print(f"No test images found in {test_data_dir}")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    
    print(f"Testing on {len(test_dataset)} images...\n")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    print(f"=====================================")
    print(f"TEST RESULTS")
    print(f"=====================================")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"=====================================")
    return accuracy

# ============ MAIN ============
if __name__ == '__main__':
    test_model('test_data', 'emotion_model.pth')