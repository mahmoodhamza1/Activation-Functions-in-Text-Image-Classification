# 🔬 Activation Functions in Text & Image Classification

This project investigates the performance of three widely used activation functions — **ReLU, Leaky ReLU, and Swish** — in deep learning models for **text classification** and **image classification** tasks.  

By implementing comparable architectures across tasks and systematically varying model depth, the project evaluates whether Swish consistently outperforms ReLU and Leaky ReLU, as suggested in prior literature.

---

## ⚙️ Methodology

### Datasets
- **Text Classification:**  
  - **20 Newsgroups dataset** (20,000 documents across 20 categories).  
  - Preprocessing: removal of headers/footers/quotes, tokenization with SpaCy, vectorization via Keras tokenizer, sequence padding to 100 tokens, label encoding.  
  - Data split: 80% training, 20% testing.  

- **Image Classification:**  
  - **CIFAR-10 dataset** (60,000 color images, 32×32 pixels, 10 balanced classes).  
  - Data split: 50,000 training, 10,000 testing.  
  - Images normalized for model training.  

---

### Models
- **Text Classification (GRU-based models):**
  - Built with increasing complexity:  
    - **Shallow:** 1 GRU layer (32 units), embedding dim = 128, dropout = 20%, learning rate = 0.01, 5 epochs.  
    - **Medium:** 2 GRU layers (64 units each), embedding dim = 129, dropout = 30%, learning rate = 0.01, 10 epochs.  
    - **Deep:** 3 GRU layers (128 units each), embedding dim = 129, dropout = 40%, learning rate = 0.001, 15 epochs.  
  - Output: Dense Softmax layer with 20 neurons (multi-class classification).  
  - Loss: categorical cross-entropy, optimizer: Adam.  

- **Image Classification (CNN-based models):**
  - Four architectures of increasing depth and complexity:  
    - **Shallow:** 2 convolutional layers (32, 64 filters), BatchNorm, MaxPooling, Dropout (20%), GAP, Dense 128 → Softmax.  
    - **Medium:** Adds a 3rd conv layer (128 filters), Dropout (30%), Dense 256.  
    - **Deep:** 4 conv layers (32–256 filters), Dropout (40%), Dense 512.  
    - **Deeper:** 6 conv layers (up to 512 filters), Dropout (50%), Dense 1024.  
  - Training: 30 epochs with early stopping, batch size 128.  
  - Loss: categorical cross-entropy, optimizer: Adam.  

---

### Experiment Setup
- Each configuration was trained separately with **ReLU**, **Leaky ReLU**, and **Swish** as activation functions.  
- Accuracy and loss were tracked on both training and validation sets.  
- Predictions were evaluated quantitatively (accuracy, loss) and qualitatively (sample predictions vs. true labels).  
- Regularization (Dropout, BatchNorm, GAP) was tuned by model depth to mitigate overfitting.  

---

## 📊 Results

### Text Classification (GRU, 20 Newsgroups)

| Activation | Shallow | Medium | Deep |
|------------|---------|--------|------|
| **ReLU**   | 31.2%   | 42.5%  | 26.5% |
| **Leaky ReLU** | 30.4% | 43.4% | 32.6% |
| **Swish**  | 32.8%   | 39.8%  | 34.2% |

- **Best performance:** Medium configuration, ~43–44% accuracy (ReLU/Leaky ReLU).  
- **Swish:** Generalized slightly better (validation curves more stable), but accuracy gains were modest.  
- **Observation:** All models showed overfitting beyond ~3rd epoch, validation accuracy plateaued ~45%.  

---

### Image Classification (CNN, CIFAR-10)

| Activation | Shallow | Medium | Deep | Deeper |
|------------|---------|--------|------|--------|
| **ReLU**   | 59.9%   | 75.3%  | 76.5% | 74.8% |
| **Leaky ReLU** | 53.6% | 68.8% | 77.3% | 68.5% |
| **Swish**  | 60.0%   | 75.6%  | 79.4% | 62.5% |

- **Best performance overall:** Swish, Deep configuration → **79.4% accuracy**, outperforming ReLU/Leaky ReLU.  
- **ReLU:** Solid in shallow/medium networks; struggled in deeper ones (validation loss fluctuations, overfitting).  
- **Leaky ReLU:** Provided stability but didn’t consistently outperform ReLU.  
- **Swish:** Smoother convergence curves, fewer spikes in loss/accuracy, better generalization in deep CNNs.  

---

## 🔑 Key Insights

- **Activation choice matters:** It directly impacts stability, generalization, and maximum achievable accuracy.  
- **Swish outperformed ReLU and Leaky ReLU** in deeper architectures due to smoother gradient flow and non-monotonicity.  
- **ReLU remains efficient** and effective for shallower to moderately deep networks.  
- **Leaky ReLU improves stability** but offers limited advantages compared to Swish.  
- **Text models (GRU):** Swish generalized slightly better, but improvements were modest.  
- **Image models (CNN):** Swish consistently produced the best results in deeper architectures, confirming its suitability for complex image tasks.  

---
