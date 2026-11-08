# 🧠 MRI Brain Tumor Classification

A Streamlit web app that classifies a brain MRI image as **glioma**, **meningioma**,
**pituitary tumor**, or **no tumor**, using a fine-tuned VGG16 model.

> ⚠️ **Educational project only.** Not a medical device. Never use it for diagnosis.

## Project structure

```
├── app.py               # Streamlit app
├── train_model.ipynb    # Training + evaluation (Google Colab)
├── train_model.py       # Same as the notebook, as a script
├── requirements.txt
└── model.h5             # NOT in the repo – create it by training (see below)
```

## Getting started

```bash
pip install -r requirements.txt
```

### 1. Get the model

`model.h5` is not stored in the repo. Either:

- run `train_model.ipynb` in Google Colab (upload the dataset to
  `MyDrive/MIR Images/Training` and `.../Testing`, one sub-folder per class), then
  download the generated `model.h5` and put it next to `app.py`; or
- copy your own trained `model.h5` next to `app.py`.

### 2. Run the app

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Model details

- **Architecture:** VGG16 (ImageNet weights), last 3 layers unfrozen, then
  Flatten → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(4, softmax)
- **Input:** 224×224 RGB, pixel values scaled to 0–1
- **Class order (output index):** `glioma`, `meningioma`, `notumor`, `pituitary`
  (alphabetical, as produced by `image_dataset_from_directory`). `CLASS_LABELS`
  in `app.py` **must** match the `Class Names:` line printed by the notebook.
- **Training:** 80/20 train/validation split of the Training folder, 30 epochs, Adam (lr 1e-4)
- **Test accuracy:** ~97% on 1,311 images

| Class | Precision | Recall |
|---|---|---|
| glioma | ~0.98 | ~0.94 |
| meningioma | ~0.93 | ~0.94 |
| notumor | ~0.99 | ~1.00 |
| pituitary | ~0.97 | ~0.99 |

Most mistakes are glioma ↔ meningioma confusions (see the confusion matrix in the notebook).
Results are on a single public dataset and will likely be worse on scans from other
scanners or hospitals.

## License

MIT
