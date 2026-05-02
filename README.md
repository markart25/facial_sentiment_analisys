# Facial Sentiment Analysis

A simple PyTorch project that classifies faces as **happy** or **sad** in real time from a webcam feed. A custom CNN is trained on labelled face images, and OpenCV's Haar cascade is used to detect faces in each frame before passing them through the model.

## Features

- Binary emotion classification (happy / sad) using a custom CNN built in PyTorch
- Real-time face detection from webcam via OpenCV Haar cascades
- Confidence score displayed alongside the predicted label
- Separate script for evaluating the trained model on a held-out test set

## Project structure

```
.
├── sentiment_analysis_model.py   # Training + real-time webcam detection
├── test_model.py                 # Evaluate the trained model on test_data/
├── emotion_model.pth             # Saved model weights (created after training)
├── data/                         # Training images
│   ├── happy/                    # *.jpg images of happy faces
│   └── sad/                      # *.jpg images of sad faces
└── test_data/                    # Test images (same structure as data/)
    ├── happy/
    └── sad/
```

> Filenames may differ in your repo — adjust if needed.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (`opencv-python`)
- Pillow
- NumPy

Install with:

```bash
pip install torch torchvision opencv-python pillow numpy
```

A webcam is required for the real-time detection script.

## Dataset setup

Organise your training images into folders named after each emotion:

```
data/
├── happy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── sad/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Only `.jpg` files are picked up by the loader. Mirror the same structure under `test_data/` for evaluation.

## Usage

### 1. Train the model

In `sentiment_analysis_model.py`, uncomment the training lines in the `__main__` block:

```python
print("Training model...")
train_model('data', epochs=15, batch_size=32)
```

Then run:

```bash
python sentiment_analysis_model.py
```

This will train the CNN for 15 epochs and save the weights to `emotion_model.pth`.

### 2. Run real-time detection

With `emotion_model.pth` present, run:

```bash
python sentiment_analysis_model.py
```

A window will open showing the webcam feed with green boxes around detected faces and the predicted emotion + confidence shown next to each one. Press **`q`** to quit.

### 3. Evaluate on a test set

```bash
python test_model.py
```

This loads `emotion_model.pth`, runs it over everything in `test_data/`, and prints overall accuracy.

## Model architecture

A small convolutional network designed for 128×128 RGB inputs:

- 4 convolutional blocks: `Conv2d → ReLU → MaxPool2d`
  - Channels: 3 → 32 → 64 → 128 → 256
- Fully connected head: `Linear(256·8·8 → 512) → ReLU → Dropout(0.5) → Linear(512 → 2)`

Trained with:

- Optimiser: Adam (lr = 0.001)
- Loss: Cross-entropy
- Input normalisation: mean = std = 0.5 across all three channels
- Batch size: 32, default 15 epochs

## Notes & limitations

- Only two classes are supported (happy / sad). To extend, add folders to `data/` and update the `EMOTIONS` list in both scripts.
- Haar cascades are fast but not the most accurate face detector — consider swapping in MTCNN or a DNN-based detector for better results in difficult lighting.
- The model is intentionally lightweight and there's no validation split during training, so accuracy on unseen data will depend heavily on dataset size and diversity.

## License

MIT
