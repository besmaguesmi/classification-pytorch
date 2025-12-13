import logging
import os
import json
from datetime import datetime

import mlflow
import mlflow.pytorch

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

from src.utils import plot_loss_curves


# Enhanced logging without icons
class TrainingLogger:
    @staticmethod
    def log_epoch_progress(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, improvement=False):
        """Enhanced logging for epoch progress with visual progress bar"""

        # Progress bar
        progress = (epoch + 1) / num_epochs * 100
        bar_length = 30
        filled_length = int(bar_length * (epoch + 1) // num_epochs)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        logging.info(f"Epoch {epoch + 1:3d}/{num_epochs} [{bar}] {progress:5.1f}%")
        logging.info(f"   Train - Loss: {train_loss:.6f} | Acc: {train_acc:6.2f}%")
        logging.info(f"   Val   - Loss: {val_loss:.6f} | Acc: {val_acc:6.2f}%")

        if improvement:
            logging.info("   Validation improved - Model saved!")

    @staticmethod
    def log_training_start(run_name, train_samples, val_samples, num_epochs, use_mlflow=False):
        """Log training start information"""
        logging.info("=" * 60)
        logging.info(f"STARTING TRAINING: {run_name}")
        logging.info(f"Samples: {train_samples} train, {val_samples} validation")
        logging.info(f"Epochs: {num_epochs}")
        logging.info(f"MLflow: {'ENABLED' if use_mlflow else 'DISABLED'}")
        logging.info("=" * 60)

    @staticmethod
    def log_training_complete(best_val_loss, total_epochs):
        """Log training completion"""
        logging.info("=" * 60)
        logging.info(f"TRAINING COMPLETED")
        logging.info(f"Best validation loss: {best_val_loss:.6f}")
        logging.info(f"Total epochs trained: {total_epochs}")
        logging.info("=" * 60)


def setup_mlflow(use_mlflow=True):
    """Setup MLflow tracking only if enabled"""
    if not use_mlflow:
        logging.info("MLflow: DISABLED - Skipping MLflow setup")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pytorch-classification")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logging.info(f"MLflow Tracking URI: {tracking_uri}")
    logging.info(f"MLflow Experiment: {experiment_name}")


def train_classifier(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_dir, plot_dir, device,
                     backbone, freeze_backbone, fold=None, use_mlflow=True):
    """
    Trains a CNN for classification with optional MLflow logging

    Parameters:
    -----------
    use_mlflow : bool
        Whether to enable MLflow logging (default: True)
    """
    # Setup MLflow only if enabled
    setup_mlflow(use_mlflow)

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Training metrics
    best_val_loss = float('inf')
    counter = 0
    patience = 10
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    scaler = GradScaler()
    filename = f'cnn_{backbone}_freeze_backbone_{freeze_backbone}'

    # Learning rate schedule
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    model.to(device)

    # Run name with timestamp and fold info
    run_name = f"{backbone}_freeze_{freeze_backbone}"
    if fold is not None:
        run_name += f"_fold_{fold}"
    run_name += f"_{datetime.now().strftime('%H%M%S')}"

    # Log training start
    TrainingLogger.log_training_start(
        run_name,
        len(train_loader.dataset),
        len(val_loader.dataset),
        num_epochs,
        use_mlflow
    )

    # MLflow context manager only if MLflow is enabled
    if use_mlflow:
        run_context = mlflow.start_run(run_name=run_name)
    else:
        # Create a dummy context manager that does nothing
        from contextlib import nullcontext
        run_context = nullcontext()

    with run_context:
        if use_mlflow:
            # Log hyperparameters to MLflow
            mlflow.log_params({
                "epochs": num_epochs,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "backbone": backbone,
                "freeze_backbone": freeze_backbone,
                "batch_size": train_loader.batch_size,
                "device": str(device),
                "fold": fold if fold is not None else "none",
                "optimizer": type(optimizer).__name__,
                "criterion": type(criterion).__name__,
                "scheduler": "StepLR",
                "patience": patience
            })

            # Log dataset info to MLflow
            mlflow.log_params({
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "num_classes": len(getattr(model, 'num_classes', 'unknown'))
            })

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with mixed precision
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            # Calculate training metrics
            train_accuracy = 100 * train_correct / train_total
            average_train_loss = total_train_loss / len(train_loader)
            train_losses.append(average_train_loss)
            train_accuracies.append(train_accuracy)

            # Validation phase
            model.eval()
            total_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    val_outputs = model(images)
                    val_loss = criterion(val_outputs, labels.long())
                    total_val_loss += val_loss.item()

                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (val_predicted == labels).sum().item()

            average_val_loss = total_val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_losses.append(average_val_loss)
            val_accuracies.append(val_accuracy)

            # Log metrics to MLflow only if enabled
            if use_mlflow:
                mlflow.log_metrics({
                    "train_loss": average_train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": average_val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": scheduler.get_last_lr()[0]
                }, step=epoch)

            # Update learning rate
            scheduler.step()

            # Enhanced CMD logging (always show in console)
            improvement = average_val_loss < best_val_loss
            TrainingLogger.log_epoch_progress(
                epoch, num_epochs, average_train_loss, train_accuracy,
                average_val_loss, val_accuracy, improvement
            )

            # Early stopping and model saving
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                counter = 0

                # Save model
                if fold is not None:
                    model_filename = f"{filename}_fold_{fold}.pth"
                else:
                    model_filename = f"{filename}.pth"

                model_path = os.path.join(model_dir, model_filename)
                torch.save(model.state_dict(), model_path)

                # Log model to MLflow only if enabled
                if use_mlflow:
                    mlflow.pytorch.log_model(
                        model,
                        "model",
                        registered_model_name=f"{backbone}_classifier"
                    )

                    # Log model file as artifact
                    mlflow.log_artifact(model_path, "models")

                    # Log best metrics
                    mlflow.log_metrics({
                        "best_val_loss": best_val_loss,
                        "best_val_accuracy": val_accuracy,
                        "best_epoch": epoch + 1
                    })

            else:
                counter += 1
                if counter >= patience:
                    logging.info(f'Early stopping at epoch {epoch + 1}')
                    break

        # Log final metrics to MLflow only if enabled
        if use_mlflow:
            final_metrics = {
                "final_train_loss": average_train_loss,
                "final_train_accuracy": train_accuracy,
                "final_val_loss": average_val_loss,
                "final_val_accuracy": val_accuracy,
                "total_epochs": epoch + 1
            }
            mlflow.log_metrics(final_metrics)

        # Plot loss curves
        plot_filename = f"{filename}"
        if fold is not None:
            plot_filename += f"_fold_{fold}"

        plot_path = os.path.join(plot_dir, f"{plot_filename}_loss_curves.png")
        plot_loss_curves(train_losses, val_losses, plot_filename, plot_dir)

        # Log plot to MLflow only if enabled
        if use_mlflow and os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, "plots")

        # Save training history (always save locally)
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "best_epoch": epoch + 1 - counter if counter >= patience else epoch + 1,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": val_accuracy if average_val_loss == best_val_loss else max(val_accuracies)
        }

        history_path = os.path.join(plot_dir, f"{plot_filename}_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Log history to MLflow only if enabled
        if use_mlflow:
            mlflow.log_artifact(history_path, "training_history")

        TrainingLogger.log_training_complete(best_val_loss, epoch + 1)
        return history

    return None
