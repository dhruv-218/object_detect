# YOLOv11n Inference Pipeline – Architecture-Neutral Overview

This document describes how object detection works in a YOLO-style model (e.g., YOLOv11n), without depending on specific architectural blocks like C2PSA, PAN, or C3k2. It demonstrates the complete flow from input image to final predictions.

---

## 🚘 Example Setup

- **Input image size**: `640 × 640`
- **Classes**: `['Car', 'Pedestrian', 'Truck']`
- **Anchors per scale**: `3`
- **Feature map scales**:
  - P3 → 80 × 80 (small objects)
  - P4 → 40 × 40 (medium objects)
  - P5 → 20 × 20 (large objects)

---

## 🖼️ Step 1: Input Image and Ground Truth Labels

Say the input image contains the following objects:

```json
[
  {"class": "Car", "bbox": [120, 200, 80, 60]},
  {"class": "Pedestrian", "bbox": [300, 400, 40, 100]},
  {"class": "Truck", "bbox": [500, 100, 120, 80]}
]
```

**Bbox format**: `[x_center, y_center, width, height]` in pixel coordinates

---

## 🔄 Step 2: Forward Pass Through Network

The YOLO model processes the input image through:

1. **Backbone**: Feature extraction (e.g., CSPDarknet, EfficientNet)
2. **Neck**: Feature pyramid network for multi-scale features
3. **Head**: Detection head that outputs predictions at multiple scales

---

## 📊 Step 3: Raw Model Outputs

The model produces three output tensors for different scales:

```python
[
  output_small.shape  = (1, 3, 80, 80, 8),  # P3 - small objects
  output_medium.shape = (1, 3, 40, 40, 8),  # P4 - medium objects
  output_large.shape  = (1, 3, 20, 20, 8)   # P5 - large objects
]
```

**Output tensor format**: `(batch_size, anchors, grid_height, grid_width, predictions)`

Where `predictions = 8` contains:
- `tx, ty`: Bounding box center offsets
- `tw, th`: Bounding box width/height offsets
- `to`: Objectness score
- `tc1, tc2, tc3`: Class probabilities for each class

---

## 🧮 Step 4: Decoding Predictions

For each grid cell and anchor, convert raw outputs to actual coordinates:

```python
# Decode bounding box coordinates
x = (sigmoid(tx) + cx) / grid_size      # normalized center x
y = (sigmoid(ty) + cy) / grid_size      # normalized center y
w = anchor_w * exp(tw) / image_width    # normalized width
h = anchor_h * exp(th) / image_height   # normalized height

# Decode confidence scores
objectness = sigmoid(to)
class_probs = sigmoid([tc1, tc2, tc3])

# Final confidence per class
confidence = objectness × class_probs
```

**Where**:
- `cx, cy`: Grid cell coordinates
- `anchor_w, anchor_h`: Predefined anchor dimensions
- `sigmoid()`: Sigmoid activation function
- `exp()`: Exponential function

---

## 📈 Step 5: Confidence Calculation Example

For a detected object at grid cell (10, 15) with anchor 0:

```python
# Raw outputs
objectness = 0.8
class_probs = [0.7, 0.1, 0.2]  # [Car, Pedestrian, Truck]

# Resulting confidences:
confidence_scores = {
    "Car": 0.8 × 0.7 = 0.56,
    "Pedestrian": 0.8 × 0.1 = 0.08,
    "Truck": 0.8 × 0.2 = 0.16
}

# Predicted class: Car (highest confidence)
```

---

## 🎯 Step 6: Post-Processing

### 6.1 Confidence Thresholding
Filter out predictions with confidence below threshold (e.g., 0.25):

```python
valid_predictions = [pred for pred in predictions if pred.confidence > 0.25]
```

### 6.2 Non-Maximum Suppression (NMS)
Remove overlapping bounding boxes:

```python
selected_boxes = nms(boxes, scores, iou_threshold=0.5)
```

**NMS Process**:
1. Sort boxes by confidence score (descending)
2. Keep box with highest confidence
3. Remove boxes with IoU > threshold with kept box
4. Repeat until all boxes processed

---

## 🎉 Step 7: Final Predictions

After post-processing, the final detections might look like:

```json
[
  {
    "class": "Car",
    "confidence": 0.56,
    "bbox": [118, 198, 82, 58]
  },
  {
    "class": "Pedestrian", 
    "confidence": 0.73,
    "bbox": [300, 405, 40, 95]
  },
  {
    "class": "Truck",
    "confidence": 0.62,
    "bbox": [498, 102, 122, 78]
  }
]
```

---

## 🔧 Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `confidence_threshold` | Minimum confidence to keep detection | 0.25 |
| `iou_threshold` | IoU threshold for NMS | 0.45-0.65 |
| `max_detections` | Maximum number of detections per image | 1000 |
| `input_size` | Model input resolution | 640×640 |

---

## 📝 Pipeline Summary

1. **Input**: Resize image to 640×640
2. **Inference**: Forward pass through YOLO model
3. **Decode**: Convert raw outputs to bounding boxes and scores
4. **Filter**: Apply confidence threshold
5. **NMS**: Remove duplicate detections
6. **Output**: Final detection results

---

## 🚀 Usage Example

```python
import cv2
import numpy as np

# Load and preprocess image
image = cv2.imread('input.jpg')
image_resized = cv2.resize(image, (640, 640))
image_normalized = image_resized / 255.0

# Run inference
outputs = model.predict(image_normalized)

# Post-process
detections = postprocess(outputs, 
                        conf_threshold=0.25,
                        iou_threshold=0.45)

# Visualize results
for det in detections:
    x, y, w, h = det['bbox']
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, f"{det['class']}: {det['confidence']:.2f}", 
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

---

## 📚 Additional Resources

- [YOLOv11 Official Repository](https://github.com/ultralytics/ultralytics)
- [YOLO Paper Series](https://arxiv.org/abs/2305.08909)
- [Object Detection Fundamentals](https://arxiv.org/abs/1506.02640)

---

*This document provides a framework-agnostic understanding of YOLO inference pipeline that applies to various YOLO implementations.*
