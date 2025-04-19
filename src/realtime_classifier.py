"""
Real-Time Classification and Segmentation Pipeline

This script combines:
1. A custom classifier (MobileNetV2) trained to recognize 4 classes
2. An optional segmentation model (LRASPP MobileNetV3) for improved accuracy
3. Adaptive processing that automatically switches to segmentation when confidence is low
4. Interactive controls and visual feedback

Controls:
- Press 's' to toggle forced segmentation mode
- Press 'q' to quit
"""

import torch
import cv2
import numpy as np
from torch import nn
from torchvision import transforms, models
from PIL import Image
import time

from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

# =============================================================================
# Configuration and Initialization
# =============================================================================

# Device configuration (use GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model configuration
CLASS_NAMES = ['owner', 'pet', 'other person', 'background']
CLASSIFIER_MODEL_PATH = '../models/entity_classifier.pth'
CONFIDENCE_THRESHOLD = 0.7  # Threshold to trigger segmentation
SEGMENTATION_TIMEOUT = 5  # Seconds before auto-disabling segmentation

# Visualization colors
COLOR_HIGH_CONFIDENCE = (0, 255, 0)  # Green
COLOR_LOW_CONFIDENCE = (0, 0, 255)  # Red
COLOR_INFO = (255, 255, 0)  # Yellow
COLOR_SEGMENTATION = (0, 255, 255)  # Cyan

# Pet mask toggle
PET_MASK_ENABLED = True  # Set to False to disable generic pet detection

# =============================================================================
# Model Loading
# =============================================================================

def load_classification_model():
    """Load and configure the custom classification model"""
    # Initialize base model (MobileNetV2 without pretrained weights)
    model = models.mobilenet_v2(weights=None)

    # Modify the classifier head for our 4 classes
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASS_NAMES))
    )

    # Load trained weights
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def load_segmentation_model():
    """Load the pretrained segmentation model"""
    model = models.segmentation.lraspp_mobilenet_v3_large(
        weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    )
    model.to(DEVICE)
    model.eval()
    return model


# =============================================================================
# Image Processing Utilities
# =============================================================================

def create_classification_transform():
    """Create standard transform for classification input"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def process_segmentation_mask(mask):
    """
    Process raw segmentation mask to extract ONLY relevant classes:
    - Person (class 15)
    - (Optional: Generic cat (class 8) if we want fallback detection)

    Args:
        mask: Raw segmentation mask from model (H x W) with COCO class indices

    Returns:
        numpy.ndarray:
            Binary mask of shape (H, W) where:
            - 1 (white) = pixels belonging to selected classes (person + optional cat)
            - 0 (black) = background/other classes
            Data type: uint8 (values 0 or 1)
    """
    # STRICT MODE (only people)
    person_mask = np.array(mask == 15, dtype=np.uint8)

    # OPTIONAL: If you want to not include generic animal detection,
    # disable the PET_MASK_ENABLED flag in the Configuration and
    # Initialization section.
    pet_mask = np.array(mask == 8, dtype=np.uint8)

    combined_mask = np.clip(person_mask + pet_mask if PET_MASK_ENABLED else person_mask, 0, 1)

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    return clean_mask


# =============================================================================
# Main Processing Loop
# =============================================================================

def main():
    # Initialize models
    class_model = load_classification_model()
    seg_model = load_segmentation_model()
    transform = create_classification_transform()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State variables
    use_segmentation = False  # Adaptive segmentation flag
    force_segmentation = False  # Manual override flag
    confidence = 0.0  # Current prediction confidence
    predicted_class = "background"  # Current predicted class
    last_low_confidence_time = 0  # Timestamp of last low confidence

    # Performance tracking
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start processing timer
        start_time = time.time()

        # Convert frame to PIL Image (RGB format)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            # =================================================================
            # Classification Only Mode
            # =================================================================
            if not use_segmentation and not force_segmentation:
                # Prepare input tensor
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                # Get predictions
                outputs = class_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                new_confidence, pred_idx = torch.max(probs, 0)

                # Update state if confidence is valid
                if not torch.isnan(new_confidence):
                    confidence = new_confidence.item()
                    predicted_class = CLASS_NAMES[pred_idx]

                # Check if we need segmentation
                if confidence < CONFIDENCE_THRESHOLD:
                    use_segmentation = True
                    last_low_confidence_time = time.time()

            # =================================================================
            # Segmentation-Assisted Mode
            # =================================================================
            if use_segmentation or force_segmentation:
                # Get segmentation mask
                seg_input = transform(pil_img).unsqueeze(0).to(DEVICE)
                seg_output = seg_model(seg_input)['out']
                seg_mask = seg_output.argmax(1).squeeze().cpu().numpy()

                # Process mask to get foreground
                foreground_mask = process_segmentation_mask(seg_mask)

                # Apply mask to original image
                masked_img = np.array(pil_img.resize((224, 224))) * \
                             cv2.resize(foreground_mask, (224, 224))[..., np.newaxis]

                # Classify the masked image
                masked_pil = Image.fromarray(masked_img)
                class_input = transform(masked_pil).unsqueeze(0).to(DEVICE)
                outputs = class_model(class_input)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                seg_confidence, pred_idx = torch.max(probs, 0)

                if not torch.isnan(seg_confidence):
                    # Update if segmentation gives better confidence (or forced)
                    if force_segmentation or seg_confidence > confidence:
                        confidence = seg_confidence.item()
                        predicted_class = CLASS_NAMES[pred_idx]
                    else:
                        use_segmentation = False

                # Auto-disable segmentation after timeout
                if not force_segmentation and time.time() - last_low_confidence_time > SEGMENTATION_TIMEOUT:
                    use_segmentation = False

        # =====================================================================
        # Visualization
        # =====================================================================

        # Calculate performance metrics
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        proc_time = (time.time() - start_time) * 1000

        # Create display frame
        display_frame = frame.copy()

        # Set text color based on confidence
        text_color = COLOR_HIGH_CONFIDENCE if confidence > CONFIDENCE_THRESHOLD else COLOR_LOW_CONFIDENCE

        # Main prediction display
        cv2.putText(display_frame, f"{predicted_class} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

        # Segmentation overlay (if active)
        if use_segmentation or force_segmentation:
            overlay = frame.copy()
            seg_display = cv2.resize(foreground_mask, (frame.shape[1], frame.shape[0]))
            overlay[seg_display > 0] = COLOR_SEGMENTATION  # Highlight foreground
            display_frame = cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0)

        # Model input preview (top-right)
        preview_size = (200, 200)
        preview_x = display_frame.shape[1] - preview_size[0] - 20
        preview_y = 10

        if use_segmentation or force_segmentation:
            classifier_view = cv2.resize(masked_img, preview_size)
        else:
            classifier_view = cv2.resize(np.array(pil_img.resize((224, 224))), preview_size)
            classifier_view = cv2.cvtColor(classifier_view, cv2.COLOR_RGB2BGR)

        # Place preview in frame
        display_frame[preview_y:preview_y + preview_size[1],
        preview_x:preview_x + preview_size[0]] = classifier_view
        cv2.rectangle(display_frame,
                      (preview_x, preview_y),
                      (preview_x + preview_size[0], preview_y + preview_size[1]),
                      COLOR_SEGMENTATION, 2)
        cv2.putText(display_frame, "Model Input",
                    (preview_x + 5, preview_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SEGMENTATION, 1)

        # Performance metrics (right side)
        # FPS (Frames Per Second) and Proc (Processing time)
        metrics_y = preview_y + preview_size[1] + 30
        cv2.putText(display_frame, f"FPS: {fps:.1f}",
                    (preview_x, metrics_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_INFO, 1)
        cv2.putText(display_frame, f"Proc: {proc_time:.1f}ms",
                    (preview_x, metrics_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_INFO, 1)

        # Segmentation status
        if force_segmentation:
            seg_status = "ON (forced)"
        elif use_segmentation:
            seg_status = "ON"
        else:
            seg_status = "OFF"
        cv2.putText(display_frame, f"Seg: {seg_status}",
                    (preview_x, metrics_y + 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_SEGMENTATION, 1)

        # Display final frame
        cv2.imshow('Real-Time Classification', display_frame)

        # Handle key controls
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):  # Toggle forced segmentation
            force_segmentation = not force_segmentation
            use_segmentation = force_segmentation
            last_low_confidence_time = time.time()
        elif key & 0xFF == ord('q'):  # Quit
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()