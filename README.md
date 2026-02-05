## 🩻 MEDICAL IMAGE QUALITY ASSESSMENT (ML Pipeline)

## PROJECT OVERVIEW

This project aims to build an end-to-end Machine Learning pipeline to automatically assess the technical quality of medical images (e.g. X-ray, MRI), focusing on criteria such as sharpness, noise, contrast, and overall image integrity.

The goal is not medical diagnosis, but image quality assessment, i.e. determining whether an image is technically suitable for further analysis.


---


## MOTIVATION

In medical imaging workflows, a significant number of images are unusable due to:

motion blur

noise

poor contrast

acquisition artifacts

These low-quality images often require manual inspection or re-acquisition, increasing cost and time.

This project explores how Machine Learning and Computer Vision can help automate this first quality-control step by:

extracting objective image quality indicators

learning patterns that distinguish usable from unusable images


---


## PROBLEM DEFINITION


Given a medical image, the system outputs:

a quality score or

a binary decision:
exploitable (good quality) / non-exploitable (poor quality)

The assessment is based only on image quality, not on clinical or diagnostic content.


---

## TECHNICAL APPROACH

The project follows a progressive and explainable ML methodology.

1️⃣ Image Preprocessing

Image loading and normalization

Conversion to grayscale

Resizing for standardization

Optional denoising and contrast enhancement

2️⃣ Feature Extraction (ML Baseline)

Each image is transformed into a vector of numerical features such as:

Sharpness (variance of Laplacian)

Contrast (pixel intensity standard deviation)

Brightness

Entropy (image disorder)

These features provide an interpretable representation of image quality.

3️⃣ Machine Learning Models

Classical ML models are trained on extracted features:

Logistic Regression

Support Vector Machines

Random Forests

Models are evaluated using:

accuracy

precision / recall

confusion matrix

error analysis

4️⃣ Deep Learning Extension

A simple Convolutional Neural Network (CNN) may be implemented to:

learn features directly from raw images

compare performance and complexity with classical ML approaches


---

## EXPECTED OUTCOMES

A reproducible ML pipeline for image quality assessment

Quantitative comparison between different features and models

Analysis of model limitations and failure cases

Clear justification of design choices

---


## PROJECT STRUCTURE


medical-image-quality-assessment/

├── src/

│   ├── preprocessing.py          # common preprocessing

│   ├── features.py               # handcrafted features (baseline)

│   ├── dataset.py                # dataset loader (images + labels)

│   ├── classical_ml/             # scikit-learn pipeline

│   └── cnn/                      # CNN method (deep learning)

├── notebooks/                    # exploration & experiments

├── models/                       # saved models

├── reports/                      # plots & results

└── data/                         # raw/processed/splits


---


## TOOLS and TECHNOLOGIES

Python

OpenCV – image processing

NumPy – numerical computation

Matplotlib – visualization

scikit-learn – machine learning

PyTorch for CNN experiments

---

## SCOPE & LIMITATIONS

This project does not perform medical diagnosis.

Results depend on dataset quality and labeling.

Quality assessment criteria are technical, not clinical.


---

## AUTHOR

Abdallah-coding Computer Science student
Interested in Machine Learning Engineering, Big Data, and Applied AI
