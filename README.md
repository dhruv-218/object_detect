# Object Detection with YOLOv8

This project demonstrates object detection using a YOLOv8 neural network, trained and evaluated on a custom dataset. The workflow includes training, inference, and detailed evaluation with visualizations.

## Model Architecture
- **YOLOv8 (yolo11n.pt)**: A state-of-the-art convolutional neural network for real-time object detection.
- **Backbone**: CSPDarknet
- **Head**: YOLO detection head with anchor-free mechanism
- **Framework**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## Workflow
1. **Training**: Model is trained using `data_kaggle.yaml` for 100 epochs at 640x640 resolution.
2. **Inference**: Run predictions on test images, visualize and save results.
3. **Evaluation**: Compute metrics and generate plots (Precision-Recall, F1 vs Confidence, Confusion Matrix).

## Output & Evaluation
- **Predicted Images**: Results are saved as `predicted.jpg` and `output_with_distances.jpg`.
- **Evaluation Metrics**:
    - Precision
    - Recall
    - mAP@0.5
    - mAP@0.5:0.95
    - F1 Score
    - Confusion Matrix

### Example Output Images
| Predicted Bounding Boxes | Distance Estimation |
|-------------------------|--------------------|
| ![Prediction](predicted.jpg) | ![Distance](output_with_distances.jpg) |

### Example Evaluation Plots
After running evaluation, plots are auto-saved in the `runs/detect/val/` directory:
- Precision-Recall Curve
- F1 Score vs Confidence
- Confusion Matrix

## How to Run
1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11n.pt')
   results = model.train(data='data_kaggle.yaml', epochs=100, imgsz=640)
   ```
3. **Inference**:
   ```python
   results = model('test/images/your_image.jpg')
   results[0].show()
   results[0].save(filename='predicted.jpg')
   ```
4. **Evaluation**:
   ```python
   results = model.val(data='data_kaggle.yaml')
   results.plot()  # Saves PR curve, confusion matrix, F1 curve
   ```

## Neural Network Details
- **YOLOv8**: Efficient, anchor-free, real-time detection
- **Layers**: CSPDarknet backbone, PANet neck, YOLO head
- **Activation**: SiLU

## Evaluated Scores (Example)
```
Precision:     0.85
Recall:        0.82
mAP@0.5:       0.80
mAP@0.5:0.95:  0.65
```

## File Structure
- `object.ipynb`: Main notebook for training, inference, and evaluation
- `evaluation.py`: Script for automated evaluation and plotting
- `data_kaggle.yaml`: Dataset configuration
- `yolo11n.pt`: Pretrained YOLOv8 model
- `predicted.jpg`, `output_with_distances.jpg`: Example outputs
- `graphs/`: Directory for additional plots

## References
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLOv8 Paper](https://arxiv.org/abs/2304.00560)

---
Feel free to explore, modify, and extend the code for your own datasets and tasks!
