"""
Train a YOLOv8 object detector on the Asanyukta Kathak Mudra dataset.

USAGE
-----
1. Put this script, data.yaml, requirements.txt, check_env.py in one folder.
2. In that folder, create a subfolder called `mudra_data` and extract the
   Kaggle archive.zip into it, so you end up with:

       mudra_data/
         train/images, train/labels
         valid/images, valid/labels
         test/images,  test/labels

   Then copy the corrected data.yaml (the one next to this script) INTO
   mudra_data/, replacing the one that came with the download.

3. python3 -m venv venv && source venv/bin/activate
4. pip install -r requirements.txt
5. python check_env.py        # confirm MPS (GPU) is detected
6. python train.py

The dataset is small (~1500 images, ~60 examples/class), so this uses a
small model (yolov8n) with strong augmentation and early stopping to avoid
overfitting. Expect decent-but-not-perfect accuracy given the dataset size —
more data per class would be the biggest lever for improving results later.
"""

from ultralytics import YOLO
import torch

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")

    # yolov8n = nano, smallest/fastest variant — a sensible choice given the
    # small dataset size (bigger models overfit faster on ~1500 images).
    model = YOLO("yolov8n.pt")  # downloads pretrained COCO weights automatically

    model.train(
        data="mudra_data/data.yaml",
        epochs=150,
        patience=30,           # early stop if val mAP plateaus for 30 epochs
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project="runs",
        name="mudra_yolov8n",
        exist_ok=True,

        # Augmentation — important here since we only have ~60 imgs/class.
        # Mudras are hand shapes, so we keep flips off by default settings
        # that would mirror hand geometry in a way that could confuse
        # left/right-hand-specific mudras; adjust if that's not a concern
        # for your class set.
        degrees=15,
        translate=0.1,
        scale=0.3,
        shear=5,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.0,   # set to 0.5 if left/right mirroring is fine for your mudras
        mosaic=1.0,
        mixup=0.1,

        val=True,
        plots=True,
        save=True,
        seed=42,
    )

    # Run final evaluation on the held-out test split
    print("\n--- Evaluating on test set ---")
    metrics = model.val(data="mudra_data/data.yaml", split="test")
    print(metrics)

    print("\nDone. Best weights saved at: runs/mudra_yolov8n/weights/best.pt")


if __name__ == "__main__":
    main()