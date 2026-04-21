# Biomedical ML Projects

Two end-to-end machine learning projects on real clinical biomedical data.

## Phase 1 — ECG Arrhythmia Classifier
**Notebook:** `ECG_AFib_Classifier_Phase1.ipynb`

- **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet) — 16 records, 360 Hz, cardiologist-labeled
- **Approach:** R-peak detection (neurokit2), RR interval feature engineering, Random Forest classifier
- **Results:** 85% accuracy, 0.90 AUC
- **Stack:** Python, neurokit2, scikit-learn, pandas, numpy, matplotlib

## Phase 2 — Brain Tumor MRI Classifier
**Notebook:** `MRI_BrainTumor_Classifier_Phase2.ipynb`

- **Dataset:** Brain Tumor MRI Dataset — 7,200 MRI scans, 4 classes (glioma, meningioma, pituitary, no tumor)
- **Approach:** ResNet-18 transfer learning (ImageNet → MRI), fine-tuned with PyTorch
- **Results:** 95% test accuracy, 0.9897 AUC, 100% recall on no-tumor class
- **Stack:** Python, PyTorch, torchvision, scikit-learn, matplotlib, seaborn

## Skills Demonstrated
- Biomedical signal processing and medical image analysis
- Classical ML (Random Forest) and deep learning (CNN + transfer learning)
- Clinical evaluation: recall vs. precision tradeoffs in diagnostic contexts
- End-to-end pipeline: raw data → features → model → evaluation
