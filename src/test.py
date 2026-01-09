import logging
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support


def test_classifier(model, test_loader, plot_dir, backbone, freeze_backbone, class_names, device):
    """
    Evaluates the model on labeled data and returns comprehensive metrics for MLflow logging.

    Parameters:
    -----------
    model : nn.Module
        The trained model to evaluate or use for inference.
    test_loader : DataLoader
        DataLoader for the test dataset.
    plot_dir : str
        Directory path to save evaluation plots.
    backbone : str
        Name of the model's backbone architecture.
    freeze_backbone : bool
        Whether to freeze the backbone layers during training.
    class_names : list
        List of class names (e.g., 'sea', 'forest').
    device : torch.device
        Device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
    --------
    dict: Comprehensive test metrics and results
    """
    # Set the model to evaluation mode
    model.eval()

    # Track metrics
    correct_preds = 0
    total_samples = 0
    true_labels = []
    predictions = []
    probabilities = []
    test_loss = 0.0

    # Add loss calculation if you want it
    criterion = torch.nn.CrossEntropyLoss()

    # CUDA memory consumption (if using GPU)
    if device.type == 'cuda':
        torch.cuda.reset_max_memory_allocated(device)
        initial_memory = torch.cuda.memory_allocated(device)

    logging.info("Running inference on test dataset...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device).long()

            # Forward pass through the model
            outputs = model(images)

            # Calculate loss (optional)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Compute accuracy
            correct_preds += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

            # Collect for metrics
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

            # Progress logging every 100 batches
            if (batch_idx + 1) % 100 == 0:
                logging.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")

    # Calculate metrics
    accuracy = correct_preds / total_samples
    misclassification_rate = 1 - accuracy
    test_loss /= len(test_loader)  # Average loss

    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )

    # Per-class metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )

    # Log results
    logging.info("=" * 60)
    logging.info("TEST RESULTS SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total samples: {total_samples}")
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Misclassification rate: {misclassification_rate * 100:.2f}%")
    logging.info(f"Test loss: {test_loss:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    logging.info("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    logging.info(f"\n{cm_df}")

    # Classification report
    class_report = classification_report(true_labels, predictions,
                                        target_names=class_names,
                                        output_dict=True,
                                        zero_division=0)

    class_report_df = pd.DataFrame(class_report).transpose()
    logging.info("Detailed Classification Report:")
    logging.info(f"\n{class_report_df}")

    # Create and save plots
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title(f'Confusion Matrix - {backbone} (Freeze: {freeze_backbone})')
    cm_path = os.path.join(plot_dir, f"confusion_matrix_{backbone}_freeze_backbone_{freeze_backbone}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrix saved to: {cm_path}")

    # 2. Normalized Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title(f'Normalized Confusion Matrix - {backbone} (Freeze: {freeze_backbone})')
    cm_norm_path = os.path.join(plot_dir, f"confusion_matrix_normalized_{backbone}_freeze_backbone_{freeze_backbone}.png")
    plt.tight_layout()
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Normalized confusion matrix saved to: {cm_norm_path}")

    # Memory usage (if GPU)
    memory_used = None
    if device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated(device)
        memory_used = (final_memory - initial_memory) / (1024 ** 3)  # Convert to GB
        logging.info(f"GPU memory used during testing: {memory_used:.2f} GB")

    # Prepare comprehensive results dictionary for MLflow
    results = {
        'accuracy': accuracy,
        'misclassification_rate': misclassification_rate,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': total_samples,
        'correct_predictions': correct_preds,
        'confusion_matrix': cm.tolist(),  # Convert to list for JSON serialization
        'classification_report': class_report,
        'per_class_metrics': {
            'precision': {class_names[i]: float(class_precision[i]) for i in range(len(class_names))},
            'recall': {class_names[i]: float(class_recall[i]) for i in range(len(class_names))},
            'f1_score': {class_names[i]: float(class_f1[i]) for i in range(len(class_names))},
            'support': {class_names[i]: int(class_support[i]) for i in range(len(class_names))}
        },
        'model_metadata': {
            'backbone': backbone,
            'freeze_backbone': freeze_backbone,
            'num_classes': len(class_names),
            'class_names': class_names
        },
        'plots': {
            'confusion_matrix': cm_path,
            'confusion_matrix_normalized': cm_norm_path
        }
    }

    if memory_used is not None:
        results['gpu_memory_used_gb'] = memory_used

    # Save results to JSON file for MLflow artifacts
    results_path = os.path.join(plot_dir, f"test_results_{backbone}_freeze_backbone_{freeze_backbone}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Full test results saved to: {results_path}")

    logging.info("=" * 60)
    logging.info("TESTING COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)

    return results


def test_model_with_thresholds(model, test_loader, plot_dir, backbone, freeze_backbone,
                              class_names, device, min_accuracy=0.7, max_loss=1.0):
    """
    Enhanced testing function with performance thresholds for CI/CD pipelines.

    Parameters:
    -----------
    ... (same as test_classifier)
    min_accuracy : float
        Minimum accuracy threshold for test to pass
    max_loss : float
        Maximum loss threshold for test to pass

    Returns:
    --------
    dict: Test results with pass/fail status
    """
    results = test_classifier(model, test_loader, plot_dir, backbone,
                             freeze_backbone, class_names, device)

    # Check thresholds
    accuracy_ok = results['accuracy'] >= min_accuracy
    loss_ok = results['test_loss'] <= max_loss
    test_passed = accuracy_ok and loss_ok

    # Add threshold checking to results
    results['threshold_checks'] = {
        'min_accuracy': min_accuracy,
        'max_loss': max_loss,
        'accuracy_meets_threshold': accuracy_ok,
        'loss_meets_threshold': loss_ok,
        'test_passed': test_passed
    }

    # Log threshold results
    if test_passed:
        logging.info("ALL THRESHOLDS MET - TEST PASSED")
    else:
        logging.error("SOME THRESHOLDS NOT MET - TEST FAILED")

    if not accuracy_ok:
        logging.error(f"Accuracy {results['accuracy']:.4f} < minimum required {min_accuracy}")
    if not loss_ok:
        logging.error(f"Loss {results['test_loss']:.4f} > maximum allowed {max_loss}")

    return results
