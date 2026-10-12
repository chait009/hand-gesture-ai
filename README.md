# Real-Time Hand Gesture Recognition

A computer vision project that recognizes hand gestures from images and a live webcam feed using PyTorch, MobileNetV3, Hugging Face datasets, and OpenCV.

I built this project to practice a complete computer vision workflow: working with a large image dataset, building a reusable PyTorch data pipeline, applying transfer learning, evaluating the model, and connecting the trained model to real-time webcam inference.

The current version recognizes four gestures:

- Fist
- Like
- Peace
- Stop

## Project Highlights

- Built an end-to-end image classification pipeline in PyTorch
- Streamed selected images directly from Hugging Face instead of downloading the full dataset
- Used MobileNetV3-Small with ImageNet pretrained weights
- Implemented two-stage transfer learning and fine-tuning
- Added image augmentation to improve generalization
- Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix
- Built real-time webcam inference using OpenCV
- Added prediction smoothing to reduce unstable frame-by-frame predictions

## Architecture

```text
Hugging Face Dataset
        |
        v
Class-based Image Streaming
        |
        v
Image Preprocessing + Augmentation
        |
        v
PyTorch DataLoader
        |
        v
MobileNetV3-Small
        |
        v
Transfer Learning
        |
        v
Fine-Tuning
        |
        v
Model Evaluation
        |
        v
Real-Time OpenCV Inference
```

## Dataset

The project uses a subset of the HaGRID hand gesture dataset available on Hugging Face.

```text
ntsrigaud/hagrid-subset
```

The full dataset is much larger than what I needed for the first version of the project, so instead of downloading everything locally, the data pipeline streams only the selected gesture classes.

The current model uses:

```text
fist
like
peace
stop
```

This approach reduced local disk usage and made it easier to iterate on the training pipeline.

## Model

The classifier uses `MobileNetV3-Small` with pretrained ImageNet weights.

I chose MobileNetV3 because it provides a good balance between model size, inference speed, and image classification performance, which also makes it suitable for real-time webcam inference.

Training is done in two stages.

### Stage 1 — Classification Head Training

The pretrained feature extractor is frozen and only the classifier layers are trained.

This allows the model to use the visual features already learned from ImageNet without immediately modifying the convolutional backbone.

### Stage 2 — Fine-Tuning

The final MobileNet feature blocks are unfrozen and trained using a lower learning rate.

This allows higher-level visual features to adapt to hand gestures while keeping most of the pretrained representation intact.

## Training Approach

During development, I experimented with different dataset sizes, learning rates, augmentation settings, and fine-tuning strategies.

One of the main challenges was balancing overfitting and underfitting.

An early model reached high training accuracy but performed poorly on validation data. I adjusted the training strategy by:

- Increasing the number of training samples
- Using moderate image augmentation
- Training the classifier separately before fine-tuning
- Unfreezing only the final MobileNet blocks
- Using AdamW optimization
- Reducing the learning rate during fine-tuning
- Saving the best model based on validation performance

This improved the model's generalization on unseen test images.

## Results

The current model achieved:

**Test Accuracy: 60.42%**

**Macro F1 Score: 60.27%**

The model was evaluated on 240 test images.

| Gesture | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Fist | 55.93% | 55.00% | 55.46% |
| Like | 66.67% | 73.33% | 69.84% |
| Peace | 61.40% | 58.33% | 59.83% |
| Stop | 56.90% | 55.00% | 55.93% |

The `like` gesture currently performs best.

The confusion matrix also showed that the model still has difficulty distinguishing some `fist`, `peace`, and `stop` examples. This is an area I would improve with additional training data and hand-focused cropping.

## Confusion Matrix

The evaluation script generates:

```text
results/confusion_matrix.png
```

The confusion matrix is useful because it shows where the model is making mistakes rather than relying only on overall accuracy.

## Project Structure

```text
hand-gesture-ai/
|
├── models/
│   └── best_model.pth
|
├── results/
│   └── confusion_matrix.png
|
├── src/
│   └── gesture_ai/
│       |
│       ├── data/
│       │   ├── dataset.py
│       │   └── test_dataset.py
│       |
│       ├── models/
│       │   └── model.py
│       |
│       ├── training/
│       │   ├── train.py
│       │   └── evaluate.py
│       |
│       └── inference/
│           └── webcam.py
|
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/hand-gesture-ai.git
cd hand-gesture-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python src\gesture_ai\training\train.py
```

The best checkpoint is saved as:

```text
models/best_model.pth
```

## Evaluate the Model

Run:

```bash
python src\gesture_ai\training\evaluate.py
```

The evaluation pipeline reports:

- Accuracy
- Precision
- Recall
- F1-score
- Per-class performance
- Confusion matrix

The confusion matrix is saved under:

```text
results/confusion_matrix.png
```

## Run Real-Time Webcam Recognition

Run:

```bash
python src\gesture_ai\inference\webcam.py
```

Place your hand inside the region shown on screen.

The model predicts one of:

```text
fist
like
peace
stop
```

Press `Q` to close the application.

The webcam pipeline also averages predictions across several frames to make the displayed result more stable.

## Tech Stack

**Language**

Python

**Deep Learning**

PyTorch  
torchvision  
MobileNetV3

**Computer Vision**

OpenCV  
Pillow

**Data**

Hugging Face Datasets  
NumPy

**Evaluation**

scikit-learn  
Matplotlib

## What I Learned

This project gave me practical experience with several problems that come up in real computer vision systems.

I worked through:

- Efficiently loading a large remote image dataset
- Building reusable PyTorch dataset and DataLoader components
- Image preprocessing and augmentation
- Transfer learning with pretrained CNN models
- Diagnosing overfitting and underfitting
- Fine-tuning only selected parts of a neural network
- Model checkpointing based on validation performance
- Evaluating multiclass models beyond accuracy
- Integrating a trained model with a webcam application
- Stabilizing real-time predictions across video frames

## Future Improvements

The current version is a working MVP. Some improvements I would make next are:

- Increase the amount of training data
- Add more gesture classes
- Detect and crop the hand before classification
- Train on a wider range of backgrounds and lighting conditions
- Experiment with EfficientNet and Vision Transformer architectures
- Add confidence calibration
- Export the model to ONNX
- Build a Streamlit demo
- Expose inference through FastAPI
- Containerize the application with Docker
- Add automated tests and CI/CD

## Current Status

The project currently supports end-to-end training, evaluation, and real-time webcam inference.

The main goal of this version was to build the complete computer vision pipeline first, then use the evaluation results to guide future improvements.
