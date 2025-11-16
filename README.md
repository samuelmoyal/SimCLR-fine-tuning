# One-Shot Learning for Plan Symbols using SimCLR

This repository contains a complete Jupyter notebook demonstrating how to train a **SimCLR** model to learn robust feature representations of architectural plan symbols for **one-shot learning**.  
The goal is to obtain a feature extractor capable of recognizing a new unseen symbol from a *single* example.

---

## 📌 Project Overview

The notebook walks through the full pipeline:

### 1. **Dataset Preprocessing**
- Image cleaning and formatting.
- **Automatic cropping** to isolate individual symbols.
- A full SimCLR-style augmentation pipeline:
  - Random resized crops  
  - Color Jitter  
  - Gaussian Blur  
  - Horizontal flips  
  - Normalization  

![Plan](images/preprocessing/image_plan.png)
![Image cropped 1](images/preprocessing/image_cropped_1.png)
![Image cropped 2](images/preprocessing/image_cropped_2.png)
![Data augmentation](images/preprocessing/data_augmentation.png)




These augmentations are designed to enforce invariance to contrast, blur, and spatial transformations.



### 2. **Model Architecture**
- Backbone: **ResNet** (18 or 50).
- Added **MLP projection head**, as required by the SimCLR contrastive framework.
- After training, the projection head is discarded and the backbone becomes a **feature extractor** for one-shot learning.

### 3. **Contrastive Loss: NT-Xent (InfoNCE)**
The notebook implements the full SimCLR loss:
- Positive pairs: two augmentations of the same image  
- Negative pairs: all other images in the batch  
- Temperature scaling  
- Normalized dot-product similarity  

### 4. **Training Experiments**

Several configurations were tested:

#### ⭐ Model comparisons
- ResNet50  
- ResNet18 baseline  
- ResNet18 with layer1-3 frozen: best results


#### ⭐ Hyperparameters explored
- Learning rate  
- Weight decay  
- Batch size  
- Scheduler strategies (cosine annealing, warm restarts…)
- Temperature
- Output dimensions

#### 📈 Example Training Curves

> Replace these image paths with your actual plot screenshots.

![Training Curve with train on layer4, temperature=0.5](images/plot_loss_curves/layer4only_loss_curve_adamw_scheduler_model=resnet18_out128_lr0.0001_temp0.5_w1e-05_batch64.png)
![Training Curve with train on layer4, temperature=0.6](images/plot_loss_curves/layer4only_loss_curve_adamw_scheduler_model=resnet18_out128_lr0.0001_temp0.6_w1e-06_batch64.png)


### 🔍 Weight Change Visualization (Layer 4 Analysis)

![Visualization of some weights in layer 4](images/weight_visualization/weight_visualization.png)

- Confirm that **only unfrozen layers** are modified  
- Detect potential training instabilities  
- Assess whether fine-tuning is gentle or overly disruptive


### 5. **One-Shot Evaluation Protocol**
The notebook implements a lightweight evaluation pipeline:

1. Extract embeddings from the trained backbone.  
2. Build a **support set**: one example per class.  
3. For each query image:
   - Compute its embedding
   - Find the nearest support embedding  
   - Assign the support label  

### Example t-SNE Visualization

![t-SNE Embeddings on validation set before training](images/embeddingd/Embeddings_on_validation_set_before_training.png)
![t-SNE Embeddings on validation set after training](Embeddings_on_validation_set_after_training.png)

This gives a qualitative view of how well classes separate in the learned space.

### 6. **Conclusion**
The notebook identifies training configurations that produce strong, contrast-invariant embeddings suitable for one-shot symbol recognition.  
It also highlights limitations and potential future improvements (e.g., larger batch sizes, longer training, stronger augmentations).

---

## 📄 Repository Contents

- `one_shot_simclr.ipynb` — End-to-end notebook including:
  - Data preprocessing pipeline  
  - Model construction  
  - Contrastive learning loop  
  - Evaluation on one-shot classification  

⚠️ **This repository does NOT include:**
- Model weights  
- Datasets  

This keeps the repository lightweight and avoids storing large binaries.

---

## 🚀 Usage

### Install dependencies
```bash
pip install torch torchvision numpy matplotlib scikit-learn
