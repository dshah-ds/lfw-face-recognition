# 🧑 Face Recognition on LFW: From Classical CV to Transfer Learning

Who is in this photo? Three generations of computer vision answer the same question on the same faces, scored under the same rules: hand-crafted HOG features with a linear SVM, a small CNN trained from scratch, and a fine-tuned pretrained MobileNetV2. The notebook also unpacks how Vision Transformers see a face, computing self-attention by hand.

**Interactive case study:** https://devam-shah.com/projects/face-recognition/

---

## Results on the untouched test set (454 faces, 62 people)

| Model | Test accuracy | Top-5 | Macro-F1 | Parameters | Train time | ms / image |
|---|---|---|---|---|---|---|
| Majority class (always "George W Bush") | 17.6% | – | 0.005 | 0 | – | – |
| HOG + LinearSVC | 75.6% | 85.7% | 0.675 | 312,542 | 36 s (CPU) | 9.99 |
| Baseline CNN (4 conv blocks) | 82.8% | 94.7% | 0.767 | 405,790 | 385 s (GPU) | 1.38 |
| **MobileNetV2, fine-tuned** | **86.8%** | **96.7%** | **0.808** | 2,337,406 | 212 s (GPU) | 1.98 |

95% bootstrap intervals (1,000 resamples, paired):
- Both neural models clearly beat HOG (MobileNetV2 +11.2 points, interval +7.0 to +15.6).
- MobileNetV2 vs the CNN: +4.0 points of accuracy (interval +0.2 to +7.7, only just above zero) and +0.040 macro-F1 (interval −0.012 to +0.102, **not distinguishable from zero**). Probably better, not proven better.

Recall by how much training data an identity had:

| Model | Rare (14–18 imgs) | Medium (19–26) | Common (27–371) |
|---|---|---|---|
| HOG + LinearSVC | 58.7% | 60.6% | 78.2% |
| Baseline CNN | 65.6% | 78.4% | 86.4% |
| MobileNetV2 | 75.4% | 75.3% | 91.7% |

---

## What the Notebook Covers

1. **Faces as tensors.** 3,023 colour images of 62 people, each a `(125, 94, 3)` float tensor in [0, 1]. One person (George W Bush, 530 images) is 17.5% of the data, so accuracy alone would mislead; macro-F1 is the main metric.
2. **Stratified 70/15/15 split** (2,116 / 453 / 454), every identity in every split.
3. **Classical OpenCV filters:** grayscale, histogram equalisation, Gaussian blur, Sobel, Canny, and why fixed thresholds fail on some faces.
4. **HOG + LinearSVC baseline:** 35,250 pixel values compressed to 5,040 gradient features.
5. **Baseline CNN** with augmentation, softened class weights, early stopping and learning-rate decay. Best epoch 126 of 141.
6. **Transfer learning:** MobileNetV2 with ImageNet weights. Head only reached 69.1% on validation; fine-tuning the top 40 layers jumped it to 87.4%.
7. **Vision Transformers:** a face cut into 48 patch tokens, self-attention computed by hand (48 × 48 matrix), and attention rollout on a pretrained ViT-Base/16 (85.8M parameters, not trained here: 2,116 images is far too few).
8. **One final test-set evaluation** with bootstrap intervals, rare-vs-common recall, confident mistakes and a cost comparison.

## Honest Limitations

- **Possible near-duplicate photos** across random splits (LFW has several shots from the same events), which may flatter every score.
- **Closed set:** the models can only pick among 62 known people and are sometimes confidently wrong (the ten worst mistakes had 93–100% confidence).
- **Single runs:** GPU training isn't exactly repeatable; differences of a point or two between models are within noise.

---

## Dataset

**Labeled Faces in the Wild (LFW)**, loaded with `sklearn.datasets.fetch_lfw_people(min_faces_per_person=20, color=True)`; it downloads on first run, nothing to add manually.
Huang, Ramesh, Berg & Learned-Miller (2007), *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*, UMass Amherst Tech Report 07-49. The images are news photographs of public figures, used here for research and education.

## How to Run

Runs best on a GPU (it was run on Google Colab with a T4).

```bash
pip install -r requirements.txt
jupyter notebook face_recognition_lfw.ipynb
```

## Tech Stack

Python · NumPy · scikit-learn · scikit-image (HOG) · OpenCV · TensorFlow / Keras · PyTorch + Hugging Face Transformers (ViT attention) · Matplotlib

---

**Devam Shah** · [devam-shah.com](https://devam-shah.com) · [LinkedIn](https://www.linkedin.com/in/devam-a-shah) · [GitHub](https://github.com/dshah-ds)
