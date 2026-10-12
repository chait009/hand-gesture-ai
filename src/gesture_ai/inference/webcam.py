import os
import sys
from collections import deque

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from src.gesture_ai.models.model import create_model


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "models/best_model.pth"

CAMERA_INDEX = 0

IMAGE_SIZE = 224

CONFIDENCE_THRESHOLD = 0.45

SMOOTHING_WINDOW = 7


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\n========================================"
)

print(
    "Real-Time Hand Gesture Recognition"
)

print(
    "========================================"
)

print(
    f"Device: {device}"
)


# =========================================================
# Check model
# =========================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"\nModel not found: {MODEL_PATH}\n"
        "Run training first."
    )


# =========================================================
# Load checkpoint
# =========================================================

print(
    "\nLoading trained model..."
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)


class_names = checkpoint[
    "class_names"
]

num_classes = checkpoint[
    "num_classes"
]


print(
    f"Classes: {class_names}"
)

print(
    f"Validation accuracy: "
    f"{checkpoint['validation_accuracy']:.2f}%"
)


# =========================================================
# Create model
# =========================================================

model = create_model(
    num_classes=num_classes,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    device
)


model.eval()


print(
    "Model loaded successfully."
)


# =========================================================
# Image preprocessing
# =========================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# =========================================================
# Prediction history
# =========================================================

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


# =========================================================
# Prediction function
# =========================================================

def predict_gesture(
    roi,
):

    # -----------------------------------------------------
    # OpenCV uses BGR
    # Convert to RGB
    # -----------------------------------------------------

    rgb_image = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB,
    )


    # -----------------------------------------------------
    # Convert to PIL
    # -----------------------------------------------------

    pil_image = Image.fromarray(
        rgb_image
    )


    # -----------------------------------------------------
    # Transform
    # -----------------------------------------------------

    image_tensor = transform(
        pil_image
    )


    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1,
        )


    probabilities = (
        probabilities
        .cpu()
        .numpy()[0]
    )


    return probabilities


# =========================================================
# Open camera
# =========================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)


if not camera.isOpened():

    raise RuntimeError(
        "\nCould not open webcam.\n"
        "Try CAMERA_INDEX = 1 if you have "
        "multiple cameras."
    )


# =========================================================
# Camera settings
# =========================================================

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280,
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720,
)


print(
    "\nCamera started."
)

print(
    "Place your hand inside the box."
)

print(
    "Press Q to exit."
)


# =========================================================
# Main loop
# =========================================================

while True:

    success, frame = (
        camera.read()
    )


    if not success:

        print(
            "Could not read camera frame."
        )

        break


    # -----------------------------------------------------
    # Mirror webcam
    # -----------------------------------------------------

    frame = cv2.flip(
        frame,
        1,
    )


    frame_height, frame_width = (
        frame.shape[:2]
    )


    # =====================================================
    # Region of interest
    # =====================================================

    roi_size = min(
        350,
        frame_height - 100,
    )


    x1 = (
        frame_width
        - roi_size
        - 50
    )

    y1 = 100

    x2 = (
        x1
        + roi_size
    )

    y2 = (
        y1
        + roi_size
    )


    # -----------------------------------------------------
    # Ensure coordinates are valid
    # -----------------------------------------------------

    x1 = max(
        0,
        x1,
    )

    y1 = max(
        0,
        y1,
    )

    x2 = min(
        frame_width,
        x2,
    )

    y2 = min(
        frame_height,
        y2,
    )


    roi = frame[
        y1:y2,
        x1:x2,
    ]


    # =====================================================
    # Predict
    # =====================================================

    if roi.size > 0:

        probabilities = (
            predict_gesture(
                roi
            )
        )


        prediction_history.append(
            probabilities
        )


        # -------------------------------------------------
        # Average recent predictions
        # -------------------------------------------------

        average_probabilities = (
            np.mean(
                np.array(
                    prediction_history
                ),
                axis=0,
            )
        )


        predicted_index = int(
            np.argmax(
                average_probabilities
            )
        )


        confidence = float(
            average_probabilities[
                predicted_index
            ]
        )


        predicted_class = (
            class_names[
                predicted_index
            ]
        )


        # =================================================
        # Confidence filtering
        # =================================================

        if (
            confidence
            >= CONFIDENCE_THRESHOLD
        ):

            display_label = (
                f"{predicted_class} "
                f"{confidence * 100:.1f}%"
            )

        else:

            display_label = (
                f"Uncertain "
                f"{confidence * 100:.1f}%"
            )


        # =================================================
        # ROI box
        # =================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )


        # =================================================
        # Main prediction
        # =================================================

        cv2.putText(
            frame,
            display_label,
            (
                x1,
                y1 - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


        # =================================================
        # Show all class probabilities
        # =================================================

        text_y = 40


        for index, class_name in enumerate(
            class_names
        ):

            class_probability = (
                average_probabilities[
                    index
                ]
            )


            probability_text = (
                f"{class_name}: "
                f"{class_probability * 100:.1f}%"
            )


            cv2.putText(
                frame,
                probability_text,
                (
                    30,
                    text_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


            text_y += 30


    # =====================================================
    # Instructions
    # =====================================================

    cv2.putText(
        frame,
        "Place hand inside box",
        (
            30,
            frame_height - 60,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


    cv2.putText(
        frame,
        "Press Q to quit",
        (
            30,
            frame_height - 25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


    # =====================================================
    # Display
    # =====================================================

    cv2.imshow(
        "Hand Gesture Recognition",
        frame,
    )


    # =====================================================
    # Quit
    # =====================================================

    key = cv2.waitKey(
        1
    ) & 0xFF


    if key == ord(
        "q"
    ):

        break


# =========================================================
# Cleanup
# =========================================================

camera.release()

cv2.destroyAllWindows()


print(
    "\nCamera closed."
)