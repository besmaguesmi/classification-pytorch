# MLOps Project Report: PyTorch Image Classification

## 1. Project Overview
This project implements an end-to-end MLOps workflow for training a ResNet18 image classifier. The system utilizes **DVC** for data versioning, **MLflow** for experiment tracking, and **GitHub Actions** for CI/CD automation.

## 2. Setup and Configuration

### 2.1 Credential Management (Security)
To ensure security, Google Cloud credentials were isolated from the codebase.
* **Step:** Generated `class-480912-2185b493e04e.json` from Google Cloud Console.
* **Protection:** Added `class-480912-2185b493e04e.json` to `.gitignore` to prevent accidental commits.
* **Verification:** Ran `git status` to confirm the file remained untracked before pushing code.

### 2.2 Data Version Control (DVC) Setup
We handled large datasets and model artifacts using DVC with a Google Drive remote.

* **Initialization:** Ran `dvc init` .
* **Remote Storage:**
    ```bash
    dvc remote add -d storage gdrive://1gMJ97yfq7SexkrRVWKbXfsT3KO5Zb82M
    dvc remote modify storage gdrive_use_service_account true
    dvc remote modify storage gdrive_service_account_json_file_path class-480912-2185b493e04e.json
    ```
* **Tracking:** Large folders (`data/`, `models/`) were ignored by Git and tracked via `.dvc` pointer files.
![alt text](image-1.png)

![alt text](image-2.png)

### 2.3 Environment & Dependency Resolution
A major challenge encountered was a DLL conflict in Anaconda

* **Root Cause:** Conflicting `libiomp5md.dll` versions in the base Conda environment.
* **Resolution:**
    1.  Uninstalled pip-based PyTorch.
    2.  Reinstalled via Conda : `conda install pytorch torchvision cpuonly -c pytorch`.
    3.  Ran: `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`.

---

## 3. Experiment Tracking (MLflow)

### 3.1 Implementation
MLflow was integrated to track hyperparameters, metrics, and model artifacts.
* **Nested Runs:** Adjusted `mlflow.start_run(nested=True)` in `train.py` to allow the main execution script to manage the parent run lifecycle without crashing during cross-validation loops.

### 3.2 UI Views
The MLflow UI (`localhost:5000`) provided real-time monitoring of the training progress.

![alt text](image.png)

---

## 4. Detailed Results Analysis

### 4.1 Performance Metrics
The ResNet18 model demonstrated exceptional performance on the test set.

* **Best Model Version:** Version 10 (Final Checkpoint)
* **Test Accuracy:** **97.14%**
* **F1 Score:** **0.97**
* **Test Loss:** **0.04**
* **Test precision:** **0.97**
* **Test recall:** **0.97**

![alt text](image-3.png)

### 4.2 Stability & Convergence
* **Loss Convergence:** The Test Loss dropped to **0.04**, indicating the model successfully minimized the error function without getting stuck in local minima.
* **Stability:** The high F1 score (0.97) matches the accuracy, proving the model is **stable across classes** (it is not just guessing the most common label).

### 4.3 Why ResNet18?
ResNet18 was chosen as the architecture. It performed best because:
1.  **Residual Connections:** Allowed for deep training without the vanishing gradient problem.
2.  **Pre-training:** Leveraging transfer learning allowed the model to converge rapidly even with a limited dataset, reaching >97% accuracy in fewer epochs.

### 4.4 Benchmarking Discussion

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Precision** | High | Low false positive rate. |
| **Recall** | High | Low false negative rate (missed detections). |
| **Inference** | Fast | ResNet18 is lightweight enough for real-time CPU inference. |

---
