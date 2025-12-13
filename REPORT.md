# DVC + MLflow Report (PyTorch Classification)

**Author:** Neji Abderrahim – CI2

---

## 1) Context & Objective

End-to-end image classification in PyTorch with MLflow experiment tracking and DVC data/model versioning.

## 2) Environment

- Virtual env: `.venv`
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```
- Quick check:
  ```bash
  python -c "import torch, mlflow; print(torch.__version__, mlflow.__version__)"
  ```

## 3) Data & DVC

- Remote: `gdrive_remote` (Google Drive)
- Pull data:
  ```bash
  python -m dvc pull
  ```
- Expected layout:
  ```text
  data/
    train/<class1|class2|…>/
    test/<class1|class2|…>/
  ```
- Track updates:
  ```bash
  python -m dvc add data
  python -m dvc push
  ```

## 4) MLflow

- UI:
  ```bash
  mlflow ui --host 0.0.0.0 --port 5000
  ```
  http://localhost:5000
- Experiment: `pytorch-classification`

## 5) Training

- Command:
  ```bash
  python main.py --mode train --data_path data/train --use_mlflow
  ```
- Models produced:
  ```text
  models/cnn_resnet18_freeze_backbone_True_fold_0..4.pth
  ```

## 6) Testing / Evaluation

- Example (fold 3):
  ```bash
  python main.py --mode test \
    --data_path data/test \
    --model_path models/cnn_resnet18_freeze_backbone_True_fold_3.pth \
    --use_mlflow
  ```

## 7) Results & Analysis

- Best run (by `best_val_accuracy`):
  - `run_id`: `408acb4f6a014991b3624747f9e48233`
  - `fold`: 3
  - `best_val_accuracy`: **75.0**
  - `artifact_uri`: `mlflow-artifacts:/1/408acb4f6a014991b3624747f9e48233/artifacts`
- Best val accuracy per fold:
  - Fold 3: 75.0
  - Fold 4: 50.0
  - Fold 1: 50.0
  - Fold 0: 50.0
  - Fold 2: 25.0
- Recommended model: `models/cnn_resnet18_freeze_backbone_True_fold_3.pth`
- Observations:
  - Freezing the backbone improves stability with a small dataset.
  - Moderate LR avoids oscillations and helps convergence.
  - High variance across folds is expected with limited data.

## 8) Registry & Artifacts

- Model registry name: `resnet18_classifier` (versions 1..4).
- Local artifacts: `mlartifacts/1/<run_id>/artifacts/...` (models, plots, histories, reports).

## 9) Git & CI

- Typical commit:
  ```bash
  git add data.dvc .gitignore src/datasets.py train.py
  git commit -m "Auto-detect classes; add MLflow runs"
  ```

## 10) Screenshots

- Added to `./screenshots/`
