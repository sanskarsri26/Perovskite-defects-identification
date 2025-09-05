# Image Clustering Using VGG16 + K-Means

Cluster images into meaningful groups using deep features extracted from a pre-trained VGG16 model, then copy/rename clustered images into an output directory for quick review.

> **Dataset notice (NDA):** Example images and the lab-generated dataset are **not** included in this repository due to NDA restrictions. Please use your own images to run the pipeline.

---

## Highlights

* **Vision Transformer + K-Means (separate research track):**

  * Trained a ViT with K-Means on a lab-generated dataset of **30K perovskite solar cell images** (Intel collaboration).
  * **+40%** improvement in defect detection accuracy vs. manual inspection across **cracks, voids, and pinholes**.
  * Data augmentation + hyperparameter tuning delivered a **+30% precision** boost, enabling partial automation of QA workflows.

> The README below documents the VGG16 + K-Means clustering script. The ViT results summarize an associated research track in this project.

---

## How It Works

1. **Image feature extraction**

   * Loads **VGG16** pre-trained on ImageNet **without** the top classification layers.
   * Resizes each image to **224×224**, preprocesses for VGG16, and extracts a pooled feature vector.

2. **Clustering**

   * Runs **K-Means** on the feature vectors to form `number_clusters` groups.

3. **Copying & renaming**

   * Copies each image into the target directory and renames it as:

     ```
     cluster_<label>_<index>.jpg
     ```

---

## Requirements

* Python 3.9+ recommended
* Libraries:

  * `keras` / `tensorflow` (Keras backend)
  * `scikit-learn`
  * `numpy`
  * `pillow` (PIL)
  * Standard libs: `glob`, `os`, `shutil`, `pathlib`

Install with:

```bash
pip install keras tensorflow scikit-learn numpy pillow
```

---

## Quick Start

1. **Prepare directories**

   * Put your input **.jpg** images in a folder, e.g. `C:/indir/IMG_PERO`
   * Choose an output folder, e.g. `C:/outdir/IMG_PERO_OUTPUT`

2. **Set parameters in the script**

   * `imdir`: path to input images
   * `targetdir`: path to write clustered outputs
   * `number_clusters`: number of clusters to produce

3. **Run**

   ```bash
   python cluster_images.py
   ```

---

## Example Configuration

```python
# cluster_images.py (minimal example)

from pathlib import Path
import os, shutil, glob
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array

# --- User config ---
imdir = r"C:/indir/IMG_PERO"
targetdir = r"C:/outdir/IMG_PERO_OUTPUT"
number_clusters = 10
# -------------------

os.makedirs(targetdir, exist_ok=True)

# Load VGG16 backbone (no top, global average pooling)
model = VGG16(weights="imagenet", include_top=False, pooling="avg")

features = []
paths = []
for p in glob.glob(str(Path(imdir) / "*.jpg")):
    try:
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = img_to_array(img)[None, ...]  # shape (1, 224, 224, 3)
        arr = preprocess_input(arr)
        feat = model.predict(arr, verbose=0)
        features.append(feat.squeeze())
        paths.append(p)
    except Exception as e:
        print(f"[skip] {p} -> {e}")

if not features:
    raise SystemExit("No features extracted. Check your input directory and image format.")

X = np.vstack(features)
labels = KMeans(n_clusters=number_clusters, random_state=42, n_init="auto").fit_predict(X)

for idx, (src, lab) in enumerate(zip(paths, labels)):
    fname = f"cluster_{lab}_{idx:05d}.jpg"
    dst = str(Path(targetdir) / fname)
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"[copy-skip] {src} -> {e}")

print(f"Done. Wrote clustered images to: {targetdir}")
```

---

## Usage Notes

* Input images should be **`.jpg`**. (Non-JPG images are ignored in the example script.)
* Failed image reads are **skipped**; the script continues.
* The output directory is created if it doesn’t exist.
* **Runtime:** K-Means can be slow for large datasets or high cluster counts—start with a modest `number_clusters`.

---

## Example

```python
imdir = 'C:/indir/IMG_PERO'
targetdir = 'C:/outdir/IMG_PERO_OUTPUT'
number_clusters = 10
```

This configuration clusters 100+ images (or however many are present) from `IMG_PERO` into **10** groups and writes renamed copies to `IMG_PERO_OUTPUT`.

---

## Troubleshooting

* **No images found**: Confirm `imdir` is correct and contains `.jpg` files.
* **Memory issues** with very large datasets: Consider batching feature extraction and saving features to disk (e.g., `.npy`) before clustering.
* **Mixed formats**: Extend the glob to include `*.png`, etc., and adjust preprocessing if needed.

---

## NDA & Data Access

* **Images are not provided** in this repository due to confidentiality agreements (NDA).
* The perovskite solar cell dataset (30K images; Intel collaboration) is restricted.
  Use your own data to reproduce clustering or contact project maintainers for potential data-sharing terms (if applicable).

---

## Acknowledgments

* **VGG16** backbone from `keras.applications` (ImageNet weights)
* **TensorFlow/Keras**, **scikit-learn**, **NumPy**, **Pillow**

---

## License

Add your chosen license here (e.g., MIT).
