# Kathak Mudra Detection using YOLOv8

A computer vision project that trains a **YOLOv8 object detection model** to recognize **26 Asamyukta (single-hand) Kathak mudras** from images.

The model is trained on an annotated dataset of Kathak hand gestures and can be used as the foundation for real-time mudra recognition applications, educational tools, or dance-assistance systems.

---

## Overview

Indian classical dance relies heavily on hand gestures (mudras) to communicate meaning, emotion, and narrative. This project explores the use of modern object detection techniques to automatically identify these gestures from images.

The project uses **Ultralytics YOLOv8n**, a lightweight object detection model, fine-tuned on a custom dataset of Kathak mudras.

---

## Features

* Detects **26 Asamyukta Kathak mudras**
* Trained using **YOLOv8 Nano**
* Supports real-time inference from webcam or video
* Produces bounding boxes and gesture labels
* Easily deployable for desktop or web applications

---

## Dataset

The model was trained using the **Asanyukta Kathak Mudra Dataset**.

Each image contains:

* One annotated hand gesture
* Bounding box coordinates
* Corresponding mudra label

The dataset was divided into:

* Training Set
* Validation Set
* Test Set

---

## Supported Mudras

* Pataka
* Tripataka
* Ardhapataka
* Ardhachandra
* Kartarimukha
* Mayura
* Arala
* Shuktunda
* Mushti
* Shikhara
* Kapittha
* Kataka
* Soochi
* Chandrakala
* Padmakosha
* Sarpashirsha
* Mrigashirsha
* Simhamukha
* Kangula
* Alapadma
* Chatura
* Bhramara
* Hamsasya
* Hamsapaksha
* Mukula
* Tamrachuda

---

## Model

* **Architecture:** YOLOv8 Nano (YOLOv8n)
* **Framework:** Ultralytics
* **Language:** Python
* **Deep Learning Library:** PyTorch

Transfer learning was used by fine-tuning the pretrained YOLOv8n model on the custom mudra dataset.

---

## Training

Example training command:

```bash
yolo detect train \
    model=yolov8n.pt \
    data=data.yaml \
    epochs=150 \
    imgsz=640
```

The model automatically saves:

```
best.pt
last.pt
```

where:

* **best.pt** is the model with the best validation performance.
* **last.pt** is the model from the final training epoch.

---

## Evaluation Results

### Validation Set

| Metric    | Score |
| --------- | ----: |
| Precision | 0.795 |
| Recall    | 0.700 |
| mAP@50    | 0.775 |
| mAP@50-95 | 0.511 |

### Test Set

| Metric    | Score |
| --------- | ----: |
| Precision | 0.822 |
| Recall    | 0.671 |
| mAP@50    | 0.768 |
| mAP@50-95 | 0.492 |

These results demonstrate that the model performs well at identifying most mudras while leaving room for improvement on visually similar hand poses.

---

## Project Structure

```
MudraDetection/

├── mudra_data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── runs/
│   └── detect/
│       └── mudra_yolov8n/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── train.py
├── requirements.txt
└── README.md
```

---

## Running Inference

Using an image:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("image.jpg")
results[0].show()
```

Using a webcam:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(
    source=0,
    show=True
)
```

---

## Future Improvements

* Increase dataset size and diversity
* Improve performance on visually similar mudras
* Train larger YOLO models (YOLOv8s or YOLOv8m)
* Convert the model to ONNX for browser-based inference
* Deploy as a real-time educational web application
* Add multi-hand gesture detection
* Integrate gesture descriptions and learning assistance

---

## Technologies Used

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy

---

## Acknowledgements

* Ultralytics for the YOLOv8 framework.
* The creators of the Asanyukta Kathak Mudra Dataset for providing annotated training data.
* The Kathak community for preserving and documenting classical hand gestures.

---

## License

This project is intended for educational and research purposes. Please verify the licensing terms of the dataset before using it for commercial applications.
