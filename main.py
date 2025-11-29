import argparse
import logging
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from src.datasets import Dataset
from src.cnn import Classifier
from src.config import *
from train import train_classifier, setup_mlflow
from src.test import test_classifier
from src.load_ckpts import load_checkpoint
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def test_with_mlflow(model, test_loader, plots_dir, backbone, freeze_backbone, class_names, device, model_path, use_mlflow=True):
    """Test function with optional MLflow logging"""
    if not use_mlflow:
        logging.info("=" * 50)
        logging.info(f"STARTING MODEL TESTING (MLflow DISABLED)")
        logging.info(f"Model: {backbone}, Freeze: {freeze_backbone}")
        logging.info(f"Test samples: {len(test_loader.dataset)}")
        logging.info("=" * 50)

        # Run test without MLflow
        test_results = test_classifier(model, test_loader, plots_dir, backbone, freeze_backbone, class_names, device)
        return test_results

    # MLflow enabled
    setup_mlflow(use_mlflow)

    run_name = f"test_{backbone}_freeze_{freeze_backbone}_{datetime.now().strftime('%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log test parameters
        mlflow.log_params({
            "mode": "test",
            "backbone": backbone,
            "freeze_backbone": freeze_backbone,
            "test_samples": len(test_loader.dataset),
            "num_classes": len(class_names),
            "model_path": model_path,
            "device": str(device)
        })

        logging.info("=" * 50)
        logging.info(f"STARTING MODEL TESTING (MLflow ENABLED)")
        logging.info(f"Model: {backbone}, Freeze: {freeze_backbone}")
        logging.info(f"Test samples: {len(test_loader.dataset)}")
        logging.info("=" * 50)

        # Run the test
        test_results = test_classifier(model, test_loader, plots_dir, backbone, freeze_backbone, class_names, device)

        # Log test results to MLflow
        if test_results and isinstance(test_results, dict):
            # Extract main metrics
            main_metrics = {
                "test_accuracy": test_results.get('accuracy', 0) * 100,  # Convert to percentage
                "test_loss": test_results.get('test_loss', 0),
                "test_precision": test_results.get('precision', 0),
                "test_recall": test_results.get('recall', 0),
                "test_f1_score": test_results.get('f1_score', 0)
            }
            mlflow.log_metrics(main_metrics)

            logging.info("=" * 50)
            logging.info("TEST RESULTS SUMMARY")
            logging.info("=" * 50)
            for metric, value in main_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            # Check accuracy threshold
            accuracy = test_results.get('accuracy', 0)
            if accuracy >= 0.7:
                logging.info("Model meets accuracy requirements (>= 70%)")
            else:
                logging.warning("Model accuracy below 70% threshold")

        # Log the tested model
        mlflow.pytorch.log_model(model, "tested_model")

        logging.info("=" * 50)
        logging.info("TESTING COMPLETED SUCCESSFULLY")
        logging.info("=" * 50)

        return test_results


def main(args):
    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize the CNN model
    model = Classifier(len(CLASS_NAMES), backbone=BACKBONE, freeze_backbone=FREEZE_BACKBONE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logging.info(f"Using device: {device}")
    logging.info(f"Backbone: {BACKBONE}, Freeze: {FREEZE_BACKBONE}")
    logging.info(f"Number of classes: {len(CLASS_NAMES)}")
    logging.info(f"Classes: {CLASS_NAMES}")
    logging.info(f"MLflow: {'ENABLED' if args.use_mlflow else 'DISABLED'}")

    if args.mode == "train":
        # Load the entire dataset
        dataset = Dataset(root_dir=args.data_path, transform=transform, mode=args.mode)

        # Create directories for saving model and plots if they do not exist
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)

        criterion = torch.nn.CrossEntropyLoss()

        # Define K-fold cross-validation with k=5
        k_folds = 5
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Log experiment-level parameters to MLflow only if enabled
        if args.use_mlflow:
            setup_mlflow(args.use_mlflow)
            mlflow.log_param("k_folds", k_folds)
            mlflow.log_param("total_samples", len(dataset))
            mlflow.log_param("dataset_path", args.data_path)

        logging.info("=" * 50)
        logging.info(f"STARTING {k_folds}-FOLD CROSS VALIDATION")
        logging.info(f"Total dataset samples: {len(dataset)}")
        logging.info(f"Batch size: {BATCH_SIZE}")
        logging.info(f"Max epochs: {MAX_EPOCHS_NUM}")
        logging.info(f"MLflow: {'ENABLED' if args.use_mlflow else 'DISABLED'}")
        logging.info("=" * 50)

        fold_histories = []

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Print current fold
            logging.info('')
            logging.info('=' * 50)
            logging.info(f'FOLD {fold + 1}/{k_folds}')
            logging.info('=' * 50)

            # Sample elements randomly from a given list of ids, no replacement
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)

            # Define data loaders for training and validation
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)

            # Reinitialize model for each fold
            model = Classifier(len(CLASS_NAMES), backbone=BACKBONE, freeze_backbone=FREEZE_BACKBONE)
            model.to(device)

            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

            # Log fold information
            logging.info(f'''Fold {fold + 1} Details:
    Training size:   {len(train_subsampler)}
    Validation size: {len(val_subsampler)}
    Backbone:        {BACKBONE}
    Freeze Backbone: {FREEZE_BACKBONE}
    Batch size:      {BATCH_SIZE}
    Epochs:          {MAX_EPOCHS_NUM}
    Device:          {device}
    MLflow:          {'ENABLED' if args.use_mlflow else 'DISABLED'}
            ''')

            # Train the model for this fold
            fold_history = train_classifier(
                model, train_loader, val_loader, criterion, optimizer, MAX_EPOCHS_NUM,
                MODEL_DIR, PLOTS_DIR, device, BACKBONE, FREEZE_BACKBONE,
                fold=fold, use_mlflow=args.use_mlflow
            )
            fold_histories.append(fold_history)

            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Log cross-validation summary
        if fold_histories:
            avg_best_val_loss = sum(h['best_val_loss'] for h in fold_histories) / len(fold_histories)
            avg_best_val_accuracy = sum(h['best_val_accuracy'] for h in fold_histories) / len(fold_histories)

            if args.use_mlflow:
                mlflow.log_metrics({
                    "cv_avg_best_val_loss": avg_best_val_loss,
                    "cv_avg_best_val_accuracy": avg_best_val_accuracy
                })

            logging.info('')
            logging.info("=" * 50)
            logging.info("CROSS-VALIDATION SUMMARY")
            logging.info("=" * 50)
            logging.info(f'Average best val loss: {avg_best_val_loss:.6f}')
            logging.info(f'Average best val accuracy: {avg_best_val_accuracy:.2f}%')
            logging.info("TRAINING COMPLETED SUCCESSFULLY!")
            logging.info("=" * 50)

    elif args.mode == "test":
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)

        # Create the dataset for testing
        testset = Dataset(root_dir=args.data_path, transform=transform, mode=args.mode)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)

        # Load model checkpoint
        model, _, _ = load_checkpoint(model, args.model_path)
        logging.info(f"Model loaded from: {args.model_path}")

        # Perform testing with optional MLflow logging
        test_results = test_with_mlflow(
            model, test_loader, PLOTS_DIR, BACKBONE, FREEZE_BACKBONE,
            CLASS_NAMES, device, args.model_path, use_mlflow=args.use_mlflow
        )

        if test_results:
            logging.info("Test completed successfully")
        else:
            logging.error("Test failed or returned no results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Classification with Optional MLflow")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode to run: 'train' or 'test'")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--model_path", type=str, default="./models/",
                        help="Directory to save or load the model")
    parser.add_argument("--use_mlflow", action="store_true",
                        help="Enable MLflow logging (default: False)")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "test" and not os.path.exists(args.model_path):
        logging.error(f"Model path does not exist: {args.model_path}")
        exit(1)

    if not os.path.exists(args.data_path):
        logging.error(f"Data path does not exist: {args.data_path}")
        exit(1)

    main(args)
