# PPE Helmet Detection System

A deep learning-based Personal Protective Equipment (PPE) helmet detection system using Convolutional Neural Networks (CNN) and MobileNetV2 Transfer Learning.

The project detects whether a worker is wearing a safety helmet or not from images and webcam input.

---

# Features

- Custom CNN model
- MobileNetV2 transfer learning model
- Image preprocessing pipeline
- Dataset augmentation and normalization
- Model evaluation and comparison
- Confusion matrices and ROC curves
- Real-time webcam detection using OpenCV
- Organized project structure

---

# Project Structure

```text
PPE_Project/
│
├── deployment/
│   └── webcam_demo.py
│
├── evaluation/
│   └── evaluate_all_models.py
│
├── models/
│   ├── CNN_scratch.py
│   └── mobilenet_transfer.py
│
├── preprocessing/
│   ├── augmentation.py
│   ├── build_final_dataset.py
│   ├── clean_dataset.py
│   ├── crop_objects.py
│   ├── normalize_images.py
│   ├── parse_xml.py
│   ├── split_dataset.py
│   ├── validate_dataset.py
│   └── visualize_annotations.py
│
├── saved_models/
│
├── outputs/
│   ├── confusion_matrices/
│   └── graphs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib

---

# Dataset

The dataset contains two classes:

- helmet
- no_helmet

Dataset is split into:

- train
- validation
- test

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Mar0777/PPE-Helmet-Detection.git
```

Move into the project directory:

```bash
cd PPE-Helmet-Detection
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

# Training Models

## Train CNN Model

```bash
python models/CNN_scratch.py
```

## Train MobileNetV2 Model

```bash
python models/mobilenet_transfer.py
```

---

# Model Evaluation

Run full evaluation:

```bash
python evaluation/evaluate_all_models.py
```

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC Score

Generated outputs are saved inside:

```text
outputs/
```

---

# Webcam Deployment

Run real-time helmet detection:

```bash
python deployment/webcam_demo.py
```

Press:

```text
Q
```

to quit the webcam window.

---

# Results

The MobileNetV2 transfer learning model achieved higher accuracy and better generalization compared to the CNN model built from scratch.

Example metrics achieved:

| Model | Accuracy |
|------|------|
| CNN Scratch | ~90% |
| MobileNetV2 | ~96-97% |

---

# Notes

- Trained models and datasets are excluded from GitHub due to large file sizes.
- Models can be retrained using the provided scripts.
- Outputs and evaluation graphs are generated automatically.

---

# Future Improvements

- Object detection with YOLO
- Multi-class PPE detection
- Flask or Streamlit web application
- GPU training optimization
- Live video analytics

---

# Authors

Ammar Sherif
Rahma ahmed
sondos sabri
