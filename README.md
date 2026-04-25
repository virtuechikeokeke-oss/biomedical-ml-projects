# Biomedical ML Projects

Three end-to-end machine learning projects on real clinical biomedical data.

## Phase 1 — ECG Arrhythmia Classifier
**Notebook:** `ecg-arrhythmia-classification/ECG_AFib_Classifier_Phase1.ipynb`

- **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet) — 16 records, 360 Hz, cardiologist-labeled
- **Approach:** R-peak detection (neurokit2), RR interval feature engineering, Random Forest classifier
- **Results:** 85% accuracy, 0.90 AUC
- **Stack:** Python, neurokit2, scikit-learn, pandas, numpy, matplotlib

## Phase 2 — Brain Tumor MRI Classifier
**Notebook:** `brain-tumor-mri-classification/MRI_BrainTumor_Classifier_Phase2.ipynb`

- **Dataset:** Brain Tumor MRI Dataset — 7,200 MRI scans, 4 classes (glioma, meningioma, pituitary, no tumor)
- **Approach:** ResNet-18 transfer learning (ImageNet pretrained), fine-tuned with PyTorch
- **Results:** 95% test accuracy, 0.9897 AUC, 100% recall on no-tumor class
- **Stack:** Python, PyTorch, torchvision, scikit-learn, matplotlib, seaborn

## Phase 3 — Chest X-Ray Multi-Label Classifier
**Notebook:** `chest-xray-classification/ChestXray_DenseNet121_Phase3.ipynb`

- **Dataset:** NIH ChestX-ray14 — 112,120 chest X-rays, 14 pathology labels
- **Approach:** DenseNet-121 transfer learning (ImageNet pretrained), multi-label classification with BCELoss, Grad-CAM interpretability visualizations
- **Results:** 0.842 val AUC, 0.807 test AUC — matches CheXNet benchmark (Rajpurkar et al. 2017, 0.841)
- **Stack:** Python, PyTorch, torchvision, scikit-learn, OpenCV, matplotlib

## Skills Demonstrated
- Biomedical signal processing and medical image analysis
- Classical ML (Random Forest) and deep learning (CNN + transfer learning)
- Multi-label classification with class imbalance handling
- Model interpretability with Grad-CAM — anatomically validated heatmaps
- Clinical evaluation: AUC per pathology, recall vs precision tradeoffs
- End-to-end pipeline: raw data → preprocessing → model → evaluation → visualization
