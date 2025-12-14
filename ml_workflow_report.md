# Machine Learning Workflow Report

**Course:** Machine Learning / MLOps
**Student:** Aymen Jeddou
**Instructor:** Besma Guesmi
**Project:** End-to-End ML Workflow with DVC, Google Drive, GitHub Actions & MLflow

---

## 1. Introduction

The objective of this assignment is to reproduce and fully understand an end-to-end Machine Learning (ML) workflow that ensures **reproducibility, traceability, and automation**. The workflow integrates:

- **DVC (Data Version Control)** for data and model versioning
- **Google Drive** as remote storage for large artifacts
- **GitHub Actions** for CI/CD automation
- **MLflow** for experiment tracking and model management

**Note:** All screenshots referenced in this report can be found in the `screenshots-ML` folder.

---

## 2. Environment and Tools

### 2.1 Tools Used

- Python 3.x
- Git & GitHub
- DVC + dvc-gdrive
- Google Cloud Platform (Service Accounts)
- Google Drive (Shared Folder)
- GitHub Actions
- MLflow
- PyTorch

### 2.2 Local Environment Setup

The project was cloned from a fork of the official repository and all required dependencies were installed locally.

<details>
<summary>Repository cloned and dependencies installed</summary>

```bash
git clone https://github.com/AymenJeddou/classification-pytorch
cd classification-pytorch
pip install -r requirements.txt
```

</details>

---

## 3. Google Cloud Service Account Setup

### 3.1 Project Creation

A new Google Cloud project was created to manage authentication and API access.

### 3.2 Service Account Creation

A dedicated service account was created to allow programmatic access to Google Drive via DVC.

### 3.3 Credentials Generation

A JSON key was generated for the service account. This file was stored locally as `datamanagement-481217-bf6a5e42af41.json` and excluded from version control using `.gitignore`.

### 3.4 Google Drive API Enablement

The Google Drive API was enabled through the Google Cloud Console to allow interaction between DVC and Google Drive.

---

## 4. Google Drive Configuration

### 4.1 Shared Folder Creation

A dedicated shared folder was created on Google Drive to store DVC-tracked data and models.

### 4.2 Permissions

The service account was granted **Editor** access to the shared folder.

### 4.3 Folder ID

The folder ID `0AH68ctKxj6gZUk9PVA` was extracted from the Google Drive URL and used for DVC remote configuration.

---

## 5. DVC Configuration and Data Versioning

### 5.1 DVC Initialization

DVC was initialized inside the project repository using the command:

<details>
<summary>DVC Initialization</summary>

```bash
dvc init
```

</details>

### 5.2 Data Tracking

The `data/` directory containing the dataset was added to DVC.

<details>
<summary>Data Tracking Commands</summary>

```bash
dvc add data
git add data.dvc .gitignore
git commit -m "Add dataset structure with class folders"
```

</details>

This generated the `data.dvc` file.

### 5.3 Remote Storage Setup

Google Drive was configured as the default DVC remote using the service account credentials and the shared folder ID.

<details>
<summary>Remote Storage Configuration</summary>

```bash
dvc remote add -d gdrive_remote gdrive://0AH68ctKxj6gZUk9PVA
dvc remote modify gdrive_remote gdrive_use_service_account true
dvc remote modify gdrive_remote --local gdrive_service_account_json_file_path datamanagement-481217-bf6a5e42af41.json
```

</details>

This resulted in the `.dvc/config` being updated with the remote settings.

### 5.4 Data Push

The dataset was pushed to the configured Google Drive remote.

<details>
<summary>Successful `dvc push`</summary>

```bash
dvc push
```

</details>

---

## 6. Challenges Encountered

During the implementation of the workflow, I encountered issues with the testing and model comparison stages. Specifically, I faced difficulties running the training pipeline and generating the necessary metrics for comparison. As a result, the testing and benchmarking sections could not be completed as planned.

---

## 7. Conclusion

This assignment demonstrated the importance of MLOps practices in real-world ML systems. By integrating DVC, Google Drive, GitHub Actions, and MLflow, a reproducible, automated, and scalable ML workflow was successfully implemented (up to the data versioning stage).