# 🌱 Plant Disease Detection using Deep Learning

## 🚀 Project Summary

Developed an end-to-end **Deep Learning system for plant disease classification** using real-world image data. The project demonstrates strong skills in **Computer Vision, Transfer Learning, model optimization, and deployment**.

The system is deployed as an **interactive web application**, enabling users to upload leaf images and receive real-time predictions.

---

## 🎯 Key Highlights

* Built a **multi-class image classification model (27 classes)** using transfer learning
* Improved model performance by debugging preprocessing and fine-tuning strategies
* Optimized model for deployment using **TensorFlow Lite**
* Designed and deployed a **fully functional web application using Streamlit**
* Demonstrated understanding of **ML lifecycle: data → model → evaluation → deployment**

---

## 🧠 Technical Approach

### Model

* Backbone: **MobileNetV2 (ImageNet pretrained)**
* Transfer Learning Strategy:

  * Phase 1: Frozen base model
  * Phase 2: Fine-tuned top layers
* Custom Head:

  * GlobalAveragePooling
  * Dense layers (128 → 64)
  * Dropout for regularization

---

### Data Processing

* Input size: `224 × 224`
* Normalization: `[-1, 1]` (MobileNetV2 preprocessing)
* Data Augmentation:

  * Horizontal Flip
  * Rotation
  * Zoom
  * Contrast Adjustment

---

### Training Strategy

* Loss: Sparse Categorical Crossentropy
* Optimizer: Adam
* Metrics: Accuracy, Top-3 Accuracy
* Callbacks:

  * EarlyStopping
  * ReduceLROnPlateau

---

## 📊 Results & Learnings

* Initial baseline accuracy: **~40%**
* Observed performance drop due to incorrect fine-tuning (**~10%**)
* Diagnosed and fixed:

  * Preprocessing mismatch
  * Learning rate issues
  * Data pipeline inconsistencies

💡 Key Insight: Proper preprocessing and controlled fine-tuning are critical in transfer learning.

---

## ⚙️ Deployment

* Converted model to **TensorFlow Lite (.tflite)** for efficient inference
* Built an interactive UI using **Streamlit**
* Implemented optimized inference pipeline using **TFLite Interpreter**

### 🔗 Live Demo

👉 https://plant-disease-detector-xccxud2rbt2gmdb5imvj3u.streamlit.app/

---

## 🧰 Tech Stack

**Machine Learning**

* TensorFlow / Keras
* Transfer Learning (MobileNetV2)
* TensorFlow Lite

**Development & Deployment**

* Python
* Streamlit
* NumPy, Pillow

---

## 🏗️ Project Structure

```id="d5b5k2"
plant-disease-detector/
│
├── app.py
├── plant_disease_model.tflite
├── requirements.txt
└── README.md
```

---

## 💡 Skills Demonstrated

* Deep Learning & Computer Vision
* Transfer Learning & Fine-Tuning
* Data Preprocessing & Augmentation
* Model Debugging & Optimization
* Deployment of ML models
* End-to-End ML Project Development

---

## 🚀 Future Work

* Add **Top-3 predictions with confidence visualization**
* Integrate **disease descriptions and treatment suggestions**
* Deploy scalable backend using **FastAPI**
* Build **mobile application (Android/Flutter)**
* Add **model explainability (Grad-CAM)**

---

## 👤 About Me

**Imran Shaikh**
🎓 M.S. in AI & ML
💻 Interested in Machine Learning, Deep Learning, and AI Applications

📧 imran.sha0ikh@gmail.com
🔗 https://www.linkedin.com/in/imran-shaikh-3b904b224/

---

## ⭐ Feedback

If you found this project interesting, feel free to star the repository or connect!
