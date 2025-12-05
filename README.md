# classification-pytorch
Implementation of CNN based resnet18 on Pytorch

Key Features & Proof of Execution
1. Cloud Storage & Data Versioning (DVC + Google Drive)
We utilize Google Cloud Storage (via Google Drive) to manage large datasets and model artifacts that cannot be stored in Git.

Infrastructure: Configured a Google Service Account for secure authentication.

Versioning: Data and models are versioned using DVC, ensuring reproducibility.

Proof of Execution:
<img width="1915" height="911" alt="Screenshot 2025-12-04 200016" src="https://github.com/user-attachments/assets/2d5c3f24-9f26-4d34-a223-7771c007af15" />
<img width="1483" height="724" alt="Screenshot 2025-12-04 200050" src="https://github.com/user-attachments/assets/4890c05e-9af8-4def-bfd3-9e0a19f2b94d" />
<img width="911" height="115" alt="Screenshot 2025-12-04 213129" src="https://github.com/user-attachments/assets/2247349d-7947-455c-b424-54030a87af54" />
<img width="1919" height="535" alt="Screenshot 2025-12-04 213116" src="https://github.com/user-attachments/assets/dd7b3a01-ac2e-4ce0-9893-b0ee0d1e2714" />
<img width="1917" height="190" alt="Screenshot 2025-12-04 213951" src="https://github.com/user-attachments/assets/7b67b516-9d23-4df2-a963-5e4608f2d7a9" />
<img width="1919" height="912" alt="Screenshot 2025-12-04 214021" src="https://github.com/user-attachments/assets/5958edbc-fe42-49a2-986a-0e5eb929f94e" />

2. Experiment Tracking (MLflow)
We use MLflow to track hyperparameters (learning rate, batch size), metrics (loss, accuracy), and store artifacts (trained models).

Proof of Execution:

<img width="1473" height="656" alt="Screenshot 2025-12-04 214742" src="https://github.com/user-attachments/assets/fafe047e-31a2-4af5-a647-0ee5aab7eb89" />
<img width="1919" height="960" alt="Screenshot 2025-12-04 214754" src="https://github.com/user-attachments/assets/d2e379d2-17bf-4b18-9dd1-ff0d8f0ee976" />
<img width="1919" height="956" alt="Screenshot 2025-12-04 215523" src="https://github.com/user-attachments/assets/e6d0a59b-e532-48c4-a324-e48429b7ae7c" />
<img width="1919" height="913" alt="Screenshot 2025-12-04 215640" src="https://github.com/user-attachments/assets/f9e8e5a5-aeb7-4fb5-a702-566b07e47dc7" />
<img width="1917" height="970" alt="Screenshot 2025-12-04 220933" src="https://github.com/user-attachments/assets/8ae230b0-3de1-44cc-8b73-817e09aefaab" />
<img width="1919" height="909" alt="Screenshot 2025-12-04 221446" src="https://github.com/user-attachments/assets/606f95f7-d8a9-4850-a7eb-16446d876df1" />

3. CI/CD Pipeline (GitHub Actions)
Automated workflows are set up to run tests on every push. The pipeline authenticates with Google Drive using encrypted secrets (GDRIVE_CREDENTIALS_DATA) to pull data before testing.

Proof of Execution:
<img width="1114" height="250" alt="Screenshot 2025-12-04 214311" src="https://github.com/user-attachments/assets/bee5d1ef-056b-483a-a6fd-d978ff2714d6" />
<img width="1897" height="913" alt="Screenshot 2025-12-04 214443" src="https://github.com/user-attachments/assets/a317da48-a12a-44bf-9f40-76bd2ba02011" />
<img width="1919" height="908" alt="image" src="https://github.com/user-attachments/assets/6d7b068e-6a42-4b58-b7fc-27d817a2c6b8" />
<img width="1919" height="967" alt="Screenshot 2025-12-05 011137" src="https://github.com/user-attachments/assets/dc776f76-c3aa-4f41-a67f-c5fe9b9189b7" />
<img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/067c5052-4818-4f8c-a853-b77c667bcf6b" />
