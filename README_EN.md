## MEDICAL IMAGE QUALITY ASSESSMENT (ML Pipeline)


## Introduction

The objective of this project is primarily educational. It represents a step in my learning process of Machine Learning applied to medical imaging.
***Important note: this project does not aim to detect a pathology. It constitutes an intermediate step toward a more ambitious objective: eventually developing a model capable of detecting pneumonia from a chest X-ray.***
Before reaching that stage, it is necessary for me to understand the fundamentals:
how to build a supervised dataset,
how to extract relevant features from images,
how to train a baseline model,
and how to evaluate its performance.

In this project, I therefore focus on learning the complete process of training a supervised model.
The objective is to train a simple model using manually selected features (sharpness (Laplacian variance), contrast (standard deviation), mean brightness, Shannon entropy, SNR approximation) in order to determine whether a radiograph is of good or poor technical quality.

This step allows me to understand the logic of a supervised pipeline, to analyze the behavior of evaluation metrics, and to methodologically justify the future use of a more complex model such as a CNN.
Starting with a baseline model makes it possible to evaluate whether the extracted features are already discriminative before introducing a more complex architecture.



In this README, you will find:

**1) The results of the baseline model and their interpretation.**

**2) Detailed explanations regarding the choice of the model, the selected features, the methodology followed, and the limitations of the synthetic dataset (artificially degraded images)**

---
---



## 1) Baseline results and interpretation

Model used: Logistic Regression (binary classification), here are the results:


<pre> 
confusion matrix:
[[63 12]
 [12 13]]

  classification report:
              precision    recall  f1-score   support

           0      0.840     0.840     0.840        75
           1      0.520     0.520     0.520        25 

    accuracy                          0.760       100
   macro avg      0.680     0.680     0.680       100
weighted avg      0.760     0.760     0.760       100 
  
</pre>
    

***Interpretation:***

The confusion matrix indicates that:
- 63 poor-quality images were correctly classified
- 13 good-quality images were correctly detected
- 12 poor-quality images were incorrectly considered as good
- 12 good-quality images were incorrectly classified as poor

The model achieves an overall accuracy of 76 %. However, accuracy alone can be misleading, since the dataset is imbalanced (75 poor-quality images versus 25 good-quality images).
The detailed analysis shows that:
The model correctly detects poor-quality images (class 0) with high precision and recall (0.84). On the other hand, performance is lower for good-quality images (class 1), with an f1-score of 0.52.

This means that the model has more difficulty correctly identifying good-quality images and confuses some of them with poor-quality images.
These results indicate that the extracted features contain discriminative information, but they do not allow a perfect separation of the classes.

For me, this model serves as a reference point allowing me to evaluate the future contribution of more complex models such as CNN architectures capable of capturing non-linear relationships in the data.



---

## 2) Dataset construction and methodological choices



### Data origin

The original dataset comes from Kaggle. The radiographs used are real chest X-ray images.

However, these images are all of correct technical quality. In order to train a model capable of distinguishing between good- and poor-quality images, it was necessary to have degraded examples.

---

### Artificial generation of poor-quality images

To simulate realistic acquisition defects, degradations were artificially generated using Python (OpenCV).

The applied transformations are:

- Addition of Gaussian blur (simulation of patient movement),
- Addition of random noise,
- Reduction of contrast.

For each original good-quality image, several degraded versions were created. This made it possible to construct a supervised dataset composed of two classes: good quality and poor quality.

---

### Dataset imbalance

The final dataset is imbalanced.

This imbalance directly results from the generation process: for each original image, several degraded versions were produced, which mechanically increases the proportion of poor-quality images.

This point is important because it influences the interpretation of evaluation metrics, especially accuracy.

---

### Choice of features (Feature Engineering)

The selected features are:

- Sharpness (Laplacian variance),
- Contrast (standard deviation of intensities),
- Mean brightness,
- Shannon entropy,
- Approximation of the signal-to-noise ratio (SNR).

These features were chosen because they are directly related to the technical quality of an image:

- A blurred image has fewer sharp edges.
- A low-contrast image has a more concentrated intensity distribution.
- An image that is too dark or too bright may be difficult to use.
- Entropy measures the diversity of information within the image.
- SNR estimates the proportion of noise.

The objective was to verify whether these manually defined features possess sufficient discriminative power to separate the two classes.

---

### Model choice: Logistic Regression

Logistic Regression was used as a baseline model.

This choice is justified for several reasons:

- It is a simple and interpretable model.
- It is well suited for binary classification.
- It allows a quick validation of the relevance of the extracted features.

Starting with a simple model makes it possible to establish a comparison baseline before introducing more complex architectures.

---

### Methodological limitations

This project presents several limitations:

- The degradations are artificial and may be easier to detect than real acquisition defects.
- The dataset is imbalanced.
- The labels are deterministic (created by transformation).
- Logistic Regression assumes a linear separation between classes.

These limitations must be taken into account when interpreting the results.

---

### Perspectives

The obtained results show that the extracted features possess partial discriminative power but do not allow a perfect separation of the classes.

In the context of moving toward pneumonia detection, the use of a CNN model appears relevant, since it allows the automatic learning of complex spatial representations directly from pixels.

This project therefore constitutes a methodological preliminary step before implementing a more advanced model.

---
