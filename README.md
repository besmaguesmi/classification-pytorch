# Report

#### Repo Link : "https://github.com/MiiN1136/classification-pytorch"
*Realized by : GORRAB Yasmine - CI2*

---

## 1. Introduction

This project trains an image classification model (Sea vs. Forest) using ResNet18, with experiment tracking via MLflow and data/model versioning using DVC, integrated into a GCP and GitHub Actions workflow.

---
## 2. Shared Drive Folder Configuration
* **GCP Console Access and Initialization** :
The configuration begins within the **IAM & Admin > Service Accounts** section of the Google Cloud Console.

    * The active project selected for this setup is **My first project**.

    * We've created a service account that we'll share the folder with.

        * Service Account Name: yasminegorrab

        * Service Account ID: Automatically generated
        * We've made the project publicly accessible in order to have access through the **Content Manager** new role we've created.


* **Authentication Key Generation (JSON)** :
To enable interaction between external tools and Cloud services, a private key must be generated.

    * Required steps:

        * Access the specific service account details.

        * Navigate to the "Keys" tab.

        * Select "Add Key" > "Create new key".

    * *The JSON key file is automatically downloaded upon completion of the procedure.*

*  **Google Drive Folder Preparation for DVC** :
    * Setting up remote storage requires creating a dedicated folder on Google Drive.

    * The folder must be shared with the GCP service account email address, assigning the Editor role.

    * The folder ID must be saved for the DVC remote storage configuration.

**Screenshots:**
![alt text](./images/1-1.png)
![alt text](./images/1-3.png)
![alt text](./images/1-2.png)
![alt text](./images/1-5.png)

---
## 3. Local Setup

* **Dependency Installation** :
The required libraries for Data Version Control (DVC), Google Drive integration and requirements are installed using the following commands:

```powershell
pip install dvc dvc-gdrive
pip install -r requirements.txt
```

* **DVC Initialization** :
Initialize DVC within the project directory:
```powershell
dvc init
```

* **Data Tracking Configuration** :
To transition data management from Git to DVC, the following sequence is executed:

```powershell
git rm -r --cached 'data'
git commit -m "remove data"
dvc add data
git add data.dvc
git commit -m "add dataset and dvc tracking"
```

* **DVC Remote Configuration (Google Drive)** :
Set up the Google Drive folder as the default DVC remote storage

* **Service Account Authentication**:
Configure the remote to utilize the previously generated service account credentials:
```powershell
dvc remote modify gdrive_remote gdrive_use_service_account true
dvc remote modify gdrive_remote gdrive_acknowledge_abuse true
dvc remote modify gdrive_remote --local gdrive_service_account_json_file_path credentials.json
```

* **Enabling Auto-Staging** :
Enable automatic staging of DVC-tracked changes to streamline the workflow:
```powershell
dvc config core.autostage true
```

* **Model and Large File Management** :
Heavy assets, such as models, are migrated to DVC tracking to maintain repository performance:

```powershell
dvc add models/
git rm -r --cached 'models'
git commit -m "stop model track"
```

* **Remote Data Synchronization**
Push the tracked files to the Google Drive remote:

```powershell
dvc push
```

* **CI/CD Integration and Secret Management**
    * Security Policy Enforcement : 
        * *Issue:* An initial push attempt was rejected due to a security policy detecting the GCP private key within the Git history.

        * *Resolution*: The Git history was rewritten to purge sensitive files before re-attempting the push.


```powershell
git reset --soft HEAD~5
git reset HEAD credentials.json
```

* **Ensure the key is added to .gitignore** :
    * Add-Content .gitignore "credentials.json"
```powershell
git add .
git commit -m "Clean DVC configuration (no secrets)"
git push -f
```

* **GitHub Secrets Configuration** :
The service account JSON content is added as a GitHub Secret named **GDRIVE_CREDENTIALS_DATA** to enable secure authentication within GitHub Actions.


* **MLflow Installation and Configuration** :
Installing and launching MLflow locally

```powershell
pip install mlflow mlflow[pytorch] boto3 psycopg2-binary
mlflow ui --host 0.0.0.0 --port 5000
```

* **Upstream Synchronization and Conflict Resolution**
Link the local repo to the original source to stay updated:

```powershell
git remote add upstream https://github.com/besmaguesmi/classification-pytorch.git
git fetch upstream
git checkout main
git merge upstream/main
```

* The **test.yml** workflow passed successfully, and the project built without errors.

**Screenshots:**
![alt text](./images/venv.png)
![alt text](./images/2-1.png)
![alt text](./images/2-2.png)
![alt text](./images/1-4.png)
![alt text](./images/2-3.png)
![alt text](./images/2-4.png)
![alt text](./images/2-5.png)
![alt text](./images/2-9.5.png)
![alt text](./images/2-10.png)
![alt text](./images/2-6.png)
![alt text](./images/2-11.png)
![alt text](./images/2-7.png)
![alt text](./images/2-8.png)

---
## 4. Model Training and MLflow Tracking
### Benchmarking
Model training was executed with MLflow tracking enabled to monitor performance across different hyperparameter configurations and cross-validation folds.

* **Training Phase 1: Baseline Execution (Underfitting)The initia** 
* The training session utilized default parameters to establish a performance baseline.
    * Execution Command:
    ```PowerShell
    python main.py --mode train --data_path data/train --use_mlflow```
* Configuration:
    * Folds: 5-Fold Cross-Validation
    * Epochs: 2
    * Learning Rate: 0.00001
    * Backbone: ResNet18 (Freeze = True)
    * Batch Size: 16
* Observed Metrics:
    * Training Accuracy: ~63% | Validation Accuracy: ~70%
    * Training Loss: 0.62 | Validation Loss: 0.59
    * **Analysis:** The model exhibited signs of underfitting. The high loss and low accuracy indicated that the model had not yet converged. During test evaluation, a system warning was triggered : 
    ```WARNING - Model accuracy below 70% threshold```
---
* **Training Phase 2: Optimization and Refinement**
* To resolve the underfitting observed in Phase 1, the optimizer was transitioned to *Adam* with *Weight Decay* (0.00001) to ensure stability and prevent future overfitting.
* Configuration:
    * Folds: 3-Fold Cross-Validation
    * Epochs: 5
    * Learning Rate: 0.001
    * Backbone: ResNet18 (Freeze = True)
    * Batch Size: 16
* Observed Metrics:
    * Training Accuracy: ~97% | Validation Accuracy: ~99%
    * Training Loss: 0.07 | Validation Loss: 0.04
* **Analysis:** Increasing the learning rate and the number of epochs allowed the model to reach a *performance peak*. The loss dropped significantly, demonstrating successful convergence.
---
* **Training Phase 3: Resource Efficiency Compromise**
* A third run was conducted to evaluate if increasing the batch rate and reducing epochs could maintain high performance while saving computational resources.
* Configuration:
    * Folds: 3-Fold Cross-Validation
    * Epochs: 2
    * Learning Rate: 0.01
    * Backbone: ResNet18 (Freeze = False - Unfrozen for fine-tuning)
    * Batch Size: 32
* Observed Metrics:
    * Training Accuracy: ~98% | Validation Accuracy: ~98%
    * Training Loss: 0.057 | Validation Loss: 0.023
* **Analysis:** This configuration showed *no critical loss* in performance compared to Phase 2. While the validation accuracy dropped slightly by ~1%, the model remained highly accurate. This setup represents an ideal *resource-performance compromise*, achieving near-perfect results in significantly less time.
### Comparison of Runs
* Using the MLflow Parallel Coordinates Plot and Bar Charts, the transition from high-loss baseline runs to low-loss optimized runs is clearly visible.
* Convergence: All folds in the final runs converged toward a loss below 0.06.
* Stability: The narrow gap between training and validation metrics across all folds confirms a robust, well-generalized model.
* Champion Model:
    * The model generated during Phase 2 (Fold 3) was identified as the *"Champion" model* due to its peak validation accuracy of 99% and the lowest overall error rate.
    
| Training Phase | Epochs | Learning Rate | Accuracy (Val) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Phase 1** | 2 | 10⁻⁵ | 70% | Underfitted |
| **Phase 2** | 5 | 10⁻³ | 99% | **Champion** |
| **Phase 3** | 2 | 10⁻² | 98% | Resource Optimized |

### Conclusion
The iterative adjustment of hyperparameters within MLflow allowed for a transition from a failing baseline (70% accuracy) to a highly reliable production-ready model (98-99% accuracy).

**Screenshots:**
![alt text](./images/3-1-5.png)
![alt text](./images/3-1.png)
![alt text](./images/3-2.png)
![alt text](./images/3-4.png)
![alt text](./images/3-3.png)
![alt text](./images/3-4.png)
![alt text](./images/3-5.png)
![alt text](./images/3-6.png)
![alt text](./images/3-7.png)
![alt text](./images/3-8.png)
![alt text](./images/3-9.png)
![alt text](./images/3-10.png)
![alt text](./images/3-11.png)
![alt text](./images/3-16.png)
![alt text](./images/3-17.png)
![alt text](./images/3-12.png)
![alt text](./images/3-13.png)
![alt text](./images/3-14.png)
![alt text](./images/3-15.png)

---

## 5. Phase de Test et Validation Finale

The final testing phase was executed to validate the model's performance on a completely unseen dataset using the following command:
```powerShell
python main.py --mode test --data_path data/test --model_path models/cnn_resnet18_freeze_backbone_False_fold_2.pth --use_mlflow
```

**Test Status: SUCCESS**
* **Evaluated Model:** cnn_resnet18_freeze_backbone_False_fold_2.pth
* **Global Performance**
    * Unseen Test Set: 35 total images, consisting of 15 "Sea" and 20 "Forest" samples.
    * Accuracy: 100.00%. (impossible in more complex / real-life scenarios)
    * Misclassification Rate: 0.00%.
    * Test Loss: 0.0177.
* **Analysis**: The model achieved perfect performance on the final test set. The lack of any performance degradation between the validation and testing phases confirms optimal generalization and demonstrates that the model is not overfitted.
* **Confusion Matrix and Classification Report**
    * The model correctly classified *100%* of the test samples with zero errors.
    * Correct Classifications: 35/35.
    * Errors: 0.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Sea** | 1.00 | 1.00 | 1.00 | 15.0 |
| **Forest** | 1.00 | 1.00 | 1.00 | 20.0 | 

* **MLflow Test Visualization**
    * The test run was tracked in MLflow under the name test_resnet18_freeze_False_225403.
    * Status: Finished in 12.6 seconds.
    * Logged Metrics: Accuracy (100), F1-score (1), Precision (1), and Recall (1).
* **Technical Conclusion**
    * The final model significantly exceeds performance requirements, achieving 100% accuracy and successfully passing the defined 70% accuracy threshold. The integration of DVC for data versioning and MLflow for metric tracking ensures full reproducibility and rigorous management of the machine learning lifecycle.

**Screenshots:**
![alt text](./images/final_screebshot.png)
![alt text](./images/f-2.png)





