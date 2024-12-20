import argparse
import logging
import os
import warnings

warnings.filterwarnings('ignore')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from src.datasets import Dataset
from src.cnn import Classifier
from src.config import *
from train import train_classifier
from src.test import test_classifier
from src.load_ckpts import load_checkpoint
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # MLflow URI from environment variables
    mlflow.set_tracking_uri(os.getenv("SECRET_HOST"))

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
        kfold = KFold(n_splits=k_folds, shuffle=True)

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Print current fold
            logging.info(f'FOLD {fold}')
            logging.info('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)

            # Define data loaders for training and validation
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)

            # Initialise optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

            # Log fold and training information
            logging.info(f'''Starting training for fold {fold}:
                Training size:   {len(train_subsampler)}
                Validation size: {len(val_subsampler)}
                Backbone:        {BACKBONE}
                Freeze Backbone: {FREEZE_BACKBONE}
                Batch size:      {BATCH_SIZE}
                Epochs:          {MAX_EPOCHS_NUM}
                Device:          {device}
            ''')

            # Train the model for this fold
            train_classifier(model, train_loader, val_loader, criterion, optimizer, MAX_EPOCHS_NUM, MODEL_DIR,
                             PLOTS_DIR, device, BACKBONE, FREEZE_BACKBONE)

            # Optionally clear GPU cache
            torch.cuda.empty_cache()

        logging.info('Training complete.')

    elif args.mode == "test":
        # Create the dataset for testing
        testset = Dataset(root_dir=args.data_path, transform=transform, mode=args.mode)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)

        # Load model checkpoint
        model, _, _ = load_checkpoint(model, args.model_path)
        
        # Perform testing
        test_classifier(model, test_loader, PLOTS_DIR, BACKBONE, FREEZE_BACKBONE, CLASS_NAMES, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode to run: 'train' or 'test'")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--model_path", type=str, default="./models/",
                        help="Directory to save or load the model")

    args = parser.parse_args()
    main(args)
