import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import os

#emotion labels - HAPPY AND SAD ONLY
EMOTIONS = ['happy', 'sad']
EMOTION_MAP = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# ============ DATASET CLASS ============
class FacialExpressionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        #load images from emotion folders
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

# ============ TRAINING ============
def train_model(data_dir, epochs=15, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    #load dataset
    dataset = FacialExpressionDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print("No images found! Check your data folder structure.")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    #model, loss, optimizer
    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #training loop
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'emotion_model.pth')
    print("Model saved as emotion_model.pth")
    return model

# ============ REAL-TIME DETECTION ============
def real_time_detection(model_path='emotion_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load model
    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    #face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    

    
    #video capture
    cap = cv2.VideoCapture(0)
    
    #transform for model input
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Starting live feed. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            #extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            #preprocess
            face_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_img).unsqueeze(0).to(device)
            
            # predict
            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.softmax(output, dim=1)
                emotion_idx = output.argmax(dim=1).item()
                confidence = probabilities[0][emotion_idx].item()
            
            emotion = EMOTIONS[emotion_idx]
            
            #green rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            #text label in top-right corner of box
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x+w-150, y+-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

        
        cv2.imshow('Facial Sentiment Analysis - Happy vs Sad', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ============ MAIN ============
if __name__ == '__main__':
    #train the model
    #print("Training model...")
    #train_model('data', epochs=15, batch_size=32)
    
    
    # run real-time detection
    print("\nStarting real-time detection...")
    real_time_detection('emotion_model.pth')