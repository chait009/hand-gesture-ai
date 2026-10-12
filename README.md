# Hand Gesture Recognition with PyTorch

This project is a small computer vision application that recognizes hand gestures from images and a live webcam feed.

I built it using PyTorch, MobileNetV3, Hugging Face datasets, and OpenCV. The goal was to create a complete computer vision workflow instead of only training a model in a notebook.

The current version recognizes four gestures:

- Fist
- Like
- Peace
- Stop

## What this project does

The project covers the full pipeline:

- Streams gesture images from Hugging Face
- Selects only the classes needed for training
- Applies image preprocessing and augmentation
- Uses MobileNetV3 with transfer learning
- Trains the model in two stages
- Evaluates performance using precision, recall, F1-score, and accuracy
- Generates a confusion matrix
- Runs predictions using a live webcam

## Dataset

I used a subset of the HaGRID hand gesture dataset available on Hugging Face.

Dataset:

`ntsrigaud/hagrid-subset`

Instead of downloading the full dataset, the project streams only the selected gesture classes. This helped keep disk usage manageable while still allowing the model to train on real RGB hand images.

The four classes used in this version are:

```text
fist
like
peace
stop
```

## Model

The model is based on `MobileNetV3-Small` pretrained on ImageNet.

I used transfer learning because training a vision model completely from scratch would require much more data and training time.

Training happens in two stages.

### Stage 1

The pretrained MobileNet feature extractor is frozen and only the final classifier is trained.

### Stage 2

The final MobileNet feature blocks are unfrozen and fine-tuned using a smaller learning rate.

This allows the model to keep useful general visual features while adapting to hand gesture images.

## Results

Current test accuracy:

**60.42%**

The model was evaluated on 240 test images.

| Gesture | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Fist | 55.93% | 55.00% | 55.46% |
| Like | 66.67% | 73.33% | 69.84% |
| Peace | 61.40% | 58.33% | 59.83% |
| Stop | 56.90% | 55.00% | 55.93% |

Macro F1 score:

**60.27%**

`Like` currently performs best, while the model still confuses some `fist`, `peace`, and `stop` examples.

This is the first working version of the project, so there is still room to improve generalization with more data and better hand localization.

## Confusion Matrix

The evaluation script generates a confusion matrix and saves it here:

```text
results/confusion_matrix.png
```

This helped me understand which gesture classes were being confused instead of looking only at overall accuracy.

## Project Structure

```text
hand-gesture-ai/
│
├── models/
│   └── best_model.pth
│
├── results/
│   └── confusion_matrix.png
│
├── src/
│   └── gesture_ai/
│       │
│       ├── data/
│       │   ├── dataset.py
│       │   └── test_dataset.py
│       │
│       ├── models/
│       │   └── model.py
│       │
│       ├── training/
│       │   ├── train.py
│       │   └── evaluate.py
│       │
│       └── inference/
│           └── webcam.py
│
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

```
