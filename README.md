# 🧠 Brain Tumor Detection from MRI Images

A deep learning web app that classifies brain MRI images into four categories using **VGG16 transfer learning**, built with TensorFlow/Keras and Streamlit.

> ⚠️ **Disclaimer:** This is an educational project. It is **not** a medical device and must not be used for diagnosis. Always consult a qualified medical professional.

---

## 📌 Features

- Upload an MRI image (JPG, JPEG or PNG)
- Get a predicted class with a confidence percentage
- See the probability for each of the 4 classes
- Simple, clean web interface

## 🏷️ Classes

| Label | Meaning |
|---|---|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `pituitary` | Pituitary tumor |
| `notumor` | No tumor detected |

## 🧠 How It Works

1. **Data:** Brain MRI images organized into Training and Testing folders by class.
2. **Preprocessing:** Images are resized to 128×128 and scaled to the 0–1 range. Light brightness and contrast augmentation is applied during training.
3. **Model:** VGG16 pretrained on ImageNet, with the last layers fine-tuned, followed by a small classifier head (Flatten → Dropout → Dense(128) → Dropout → Dense(4, softmax)).
4. **Training:** 5 epochs, Adam optimizer, categorical cross-entropy loss.
5. **Deployment:** The trained model (`model.h5`) is loaded in a Streamlit app for live predictions.

## 📊 Results

Evaluated on a held-out test set of 1,311 images:

- **Overall test accuracy: ~94%**
- "No tumor" images were detected with near-perfect recall
- Pituitary and meningioma classes performed strongly
- The main weakness is **glioma being confused with meningioma**, a known hard case in MRI classification

## 🗂️ Project Structure

```
tumor-classification/
├── app.py             # Streamlit web app
├── model.h5           # Trained model
├── requirements.txt   # Python dependencies
├── train_model.ipynb  # Training notebook (Google Colab)
└── README.md
```

## 🚀 How to Run

1. **Clone or download** this repository.
2. Make sure `model.h5` is in the same folder as `app.py`.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
   If `streamlit` is not recognized (Windows), use:
   ```bash
   python -m streamlit run app.py
   ```
5. Open **http://localhost:8501** in your browser and upload an MRI image.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras (VGG16)
- Streamlit
- NumPy, Pillow, h5py


## 🔮 Future Improvements

- Train for more epochs with stronger data augmentation
- Try other architectures (ResNet, EfficientNet)
- Add Grad-CAM to show which image regions influenced the prediction
- Test on external datasets for better generalization

## 📄 License

For educational purposes only.
