# Aerial Image Classification: Sea vs. Forest

**Student:** Sarra Dhouaifi 

## Project Overview
This project implements an end-to-end MLOps pipeline for binary image classification, distinguishing between aerial views of **Forests** and **Sea**.

It is built with PyTorch and leverages industry-standard tools for reproducibility and tracking:
* **DVC (Data Version Control):** Manages large datasets and versioning.
* **MLflow:** Tracks experiments, metrics, and parameters.
* **GitHub Actions (CI/CD):** Automatically tests the code and pipelines on every push.

## Key Results
The model uses a **ResNet18** backbone and achieves high accuracy on the test set.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **97.14%** |
| **Precision** | 97.32% |
| **Recall** | 97.14% |
| **F1-Score** | 97.15% |
| **Misclassification** | 2.86% (Only 1 error in 35 test images) |

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/dhouaifisarra/classification-pytorch
    cd classification-pytorch
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull the Data (DVC)**
    This project uses DVC to handle data. Pull the dataset from remote storage:
    ```bash
    dvc pull
    ```

## Usage

### 1. Training (K-Fold Cross-Validation)
To train the model using K-Fold Cross-Validation:
```bash
python main.py --mode train --data_path data/train --use_mlflow