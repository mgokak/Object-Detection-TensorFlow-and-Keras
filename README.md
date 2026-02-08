# Object Detection using TensorFlow & Keras

## Overview

This project demonstrates a complete end-to-end Object Detection workflow using TensorFlow Object Detection API and KerasCV.  
It covers all major components required in a real-world detection system:

- Data Preparation  
- Model Architecture  
- Pretrained Model Fine-tuning  
- Inference  
- Evaluation Metrics (IoU, mAP)  
- Post-processing (NMS)  
- Pipeline Configuration  

---


## 1. Data Preparation

Prepares dataset in TensorFlow Object Detection format.

### Steps
- Organize images and annotations  
- Convert annotations to TFRecord  
- Create label map  

```python
from object_detection.utils import dataset_util
tf_example = dataset_util.tf_example_from_dict(example_dict)
```

**Why?**  
TFRecord format improves training speed and is required by TFOD API.

---

## 2. Detector Architecture

Defines a neural network for feature extraction and prediction.

```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu')
])
```

**Explanation:**  
The network learns image features and predicts bounding boxes and class labels.

---

## 3. Pipeline Configuration

Controls training parameters.

Example settings:
```
num_classes: 1
batch_size: 8
learning_rate: 0.0001
```

This configuration file connects the dataset, model, and training settings.

---

## 4. Pretrained Model Fine-Tuning

Uses transfer learning to adapt a pretrained detector.

```python
model.load_weights(checkpoint_path)
model.fit(train_dataset, epochs=10)
```

**Why?**  
Fine-tuning improves performance and reduces training time.

---

## 5. Training with TensorFlow Object Detection API

```python
!python model_main_tf2.py     --pipeline_config_path=PIPELINE_CONFIG     --model_dir=MODEL_DIR
```

Runs the full training process using TFOD API.

---

## 6. Object Detection using KerasCV

Modern detection implementation using KerasCV.

```python
import keras_cv
model = keras_cv.models.YOLOV8Detector(
    num_classes=1,
    bounding_box_format="xyxy"
)
```

---

## 7. Inference using TensorFlow Hub

Loads a pretrained model for quick predictions.

```python
import tensorflow_hub as hub
detector = hub.load(model_url)
results = detector(image)
```

---

## 8. Inference with Trained Model

```python
predictions = model.predict(image)
```

Used to detect objects on new images.

---

## 9. Intersection over Union (IoU)

Measures overlap between predicted and ground truth boxes.

```python
iou = intersection_area / union_area
```

Higher IoU indicates better localization accuracy.

---

## 10. Non-Maximum Suppression (NMS)

Removes overlapping duplicate detections.

```python
selected_indices = tf.image.non_max_suppression(
    boxes, scores, max_output_size=10, iou_threshold=0.5
)
```

---

## 11. Average Precision (mAP)

Standard evaluation metric for object detection.

```python
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_true, y_scores)
```

Mean Average Precision summarizes performance across thresholds.

---

## Workflow Summary

Data Preparation → Configuration → Fine-tuning → Training → Inference → IoU → NMS → mAP

---

## Requirements

```
tensorflow
keras-cv
tensorflow-hub
numpy
matplotlib
```

---

## Author

Manasa Vijayendra Gokak  
Graduate Student – Data Science 
