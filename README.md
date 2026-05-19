# Goal
Train a model to predict whether a static T2 coronal scan slice is positive or negative for inflammation in the distal ileum - binary classification.

---

# Data
~3000 cases total (likely close to an even split).  
If the split is not even, we can use class re-weighting or sampling strategies.

---

# Augmentations
- Rotations  
- Translations  
- Brightness variation  
- Contrast variation  
- Gaussian noise  
- Zoom  

---

# Localization Strategy (YOLO Preprocessing)
We will train and compare models on both:

### 1. Non-localized images (baseline)
Train directly on full images.

### 2. Bounding-box localized images
- Train a YOLO model for distal ileum localization  
- Use bounding boxes to crop or focus on region of interest  
- Train classifier on localized images  

I expect that training after localization will improve performance, since it removes irrelevant background information. Since we are already training a YOLO model, we can reuse it as a preprocessing step for the classification pipeline. However, we will first test baseline performance using non-localized images.

---

# Training

## Transfer Learning (primary approach)
To maximize performance (depending on time and compute), we will test multiple architectures and hyperparameter variations:

### CNN-based models
- EfficientNet (B0, B2, and possibly B4)  
- ResNet (50, 101)  
- DenseNet (121, 201)  
- ConvNeXt-Tiny  
- InceptionV3  

---

## Modern Models (Vision Transformers / Hybrids)
Exploring newer architectures:

- Vision Transformer (ViT): https://arxiv.org/abs/2010.11929  
- Swin Transformer (better for small/local detail): https://arxiv.org/abs/2103.14030  
- MaxViT (CNN + Transformer hybrid): https://arxiv.org/abs/2204.01697  
- CoAtNet (CNN + Transformer hybrid): https://arxiv.org/abs/2106.04803  

---

# Metrics
We will evaluate model performance using:

- Sensitivity (Recall)  
- Specificity  
- Precision  
- F1 Score  
- AUROC  
- Accuracy  