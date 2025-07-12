import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def get_balanced_samples(x_combined, y_combined, max_samples):
    """
    Get balanced samples across all classes dynamically.
    
    Args:
        x_combined: Combined feature data
        y_combined: Combined target labels
        max_samples: Maximum total samples to return
        
    Returns:
        X_balanced, y_balanced: Balanced subset of data
    """
    unique_classes = sorted(y_combined.unique())
    n_classes = len(unique_classes)
    
    # Calculate samples per class
    samples_per_class = min(max_samples // n_classes, int(y_combined.value_counts().min()))
    
    balanced_indices = []
    
    # Sample from each class
    for class_label in unique_classes:
        class_indices = y_combined[y_combined == class_label].index.tolist()
        
        if len(class_indices) <= samples_per_class:
            sampled_indices = class_indices
        else:
            sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False).tolist()
        
        balanced_indices.extend(sampled_indices)
    
    X_balanced = x_combined.loc[balanced_indices]
    y_balanced = y_combined.loc[balanced_indices]
    
    return X_balanced, y_balanced


def plot_umap_projection(embedding, y_values, output_path=None):
    """
    Creates a UMAP projection visualization with dynamic multi-class support.
    
    Args:
        embedding: NumPy array with UMAP projection (shape [n_samples, 2])
        y_values: Array or Series with class labels
        output_path: Path where to save the image (if None, only shows the chart)
        
    Returns:
        Matplotlib Figure object
    """
    unique_classes = sorted(np.unique(y_values))
    n_classes = len(unique_classes)
    
    # Dynamic class labels and colors
    if n_classes == 2:
        # Binary classification: use meaningful labels
        class_labels = {unique_classes[0]: "Negative", unique_classes[1]: "Positive"}
        colors = {unique_classes[0]: "#1f77b4", unique_classes[1]: "#ff7f0e"}  # Blue, Orange
    else:
        # Multi-class: use specific class labels and colors for the three classes of NSTEMI
        # TODO: generalize for n classes
        class_label_mapping = {0: 'LAD', 1: 'Cx', 2: 'CDx'}
        class_color_mapping = {0: 'green', 1: 'red', 2: 'purple'}
        
        # Use specific labels and colors, fallback to generic for classes beyond 0,1,2
        class_labels = {}
        colors = {}
        for cls in unique_classes:
            if cls in class_label_mapping:
                class_labels[cls] = class_label_mapping[cls]
                colors[cls] = class_color_mapping[cls]
            else:
                # Fallback for additional classes
                class_labels[cls] = f"Class {cls}"
                colormap = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')
                colors[cls] = colormap((cls - 3) / max(n_classes-4, 1)) if cls >= 3 else 'gray'
        
    # Create the chart
    plt.figure(figsize=(10, 8), dpi=500)
    
    # Plot each class with a different color and label
    for label in unique_classes:
        mask = y_values == label
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[label]],
            label=class_labels[label],
            alpha=0.7,
            edgecolors='w',
            linewidth=0.6,
            s=60
        )
    
    # Add title and axis labels
    plt.xlabel("1st dimension", fontsize=22)
    plt.ylabel("2nd dimension", fontsize=22)
    
    # Add grid and legend with clear labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20, edgecolor='k', fancybox=True, shadow=False)
    
    # Adjust layout to avoid cut-off elements
    plt.tight_layout()
    
    # Save the image if a path is specified
    if output_path:
        plt.savefig(output_path, dpi=500, bbox_inches='tight')
        
    fig = plt.gcf()
    
    # Close the figure if it's saved (to avoid displaying it)
    if output_path:
        plt.close()
    
    return fig





def plot_confusion_matrix(cm, classifier_name, output_dir, classes=None):
    """
    Plot confusion matrix for binary or multiclass classification.
    
    Args:
        cm: Confusion matrix
        classifier_name: Name of the classifier
        output_dir: Directory to save the plot
        classes: Class labels (optional, auto-detected if None)
    
    Returns:
        str or tuple: Path(s) to saved confusion matrix plot(s)
    """
    # Auto-detect if this is binary or multiclass
    if classes is None:
        if cm.shape[0] == 2:
            classes = ['Negative', 'Positive']
        else:
            classes = [f'Class {i}' for i in range(cm.shape[0])]
    
    is_binary = len(classes) == 2
    
    if is_binary:
        # Binary classification - single plot
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes,
                    yticklabels=classes)
                    
        plt.title(f'Confusion Matrix - {classifier_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(output_dir, 'confusion_matrix.jpg')
        plt.savefig(cm_path)
        plt.close()
        return cm_path
    
    else:
        # Multiclass classification - two plots (normalized and counts)
        plt.figure(figsize=(10, 8))
        
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot the normalized confusion matrix
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        
        plt.title(f'Normalized Confusion Matrix - {classifier_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.jpg')
        plt.savefig(cm_path)
        plt.close()
        
        # Also save the raw counts version
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        
        plt.title(f'Confusion Matrix (counts) - {classifier_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_counts_path = os.path.join(output_dir, 'confusion_matrix_counts.jpg')
        plt.savefig(cm_counts_path)
        plt.close()
        
        return cm_path, cm_counts_path


def plot_auroc(y_true, y_pred_proba, classifier_name, output_dir, auroc_score=None):
    """
    Plot AUROC curve for binary or multiclass classification.
    
    Args:
        y_true: True labels (for binary: array-like, for multiclass: array-like)
        y_pred_proba: Predicted probabilities 
                     (for binary: 1D array or 2D array with shape (n_samples, 2))
                     (for multiclass: 2D array with shape (n_samples, n_classes))
        classifier_name: Name of the classifier
        output_dir: Directory to save the plot
        auroc_score: Pre-calculated AUROC score (optional, will be calculated if None)
    
    Returns:
        str: Path to saved AUROC plot
    """
    # Determine if this is binary or multiclass
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2
    
    if is_binary:
        # Binary classification
        plt.figure(figsize=(8, 6))
        
        # Handle different probability formats for binary classification
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            # If shape is (n_samples, 2), use positive class probabilities
            y_pred_proba_positive = y_pred_proba[:, 1]
        elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
            # If shape is (n_samples, 1), flatten it
            y_pred_proba_positive = y_pred_proba.ravel()
        else:
            # If shape is (n_samples,), use as is
            y_pred_proba_positive = y_pred_proba
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba_positive)
        
        # Calculate AUROC if not provided
        if auroc_score is None:
            auroc_score = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='red', label=f'AUROC = {auroc_score:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC - {classifier_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        
        auroc_path = os.path.join(output_dir, 'AUROC.jpg')
        plt.savefig(auroc_path)
        plt.close()
        
        return auroc_path
    
    else:
        # Multiclass classification
        classes = unique_classes
        
        # Binarize the output
        y_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_bin = np.hstack((1 - y_bin, y_bin))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'Class {classes[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multiclass ROC Curves (One-vs-Rest) - {classifier_name}\nOverall AUROC = {auroc_score:.3f}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        auroc_path = os.path.join(output_dir, 'AUROC.jpg')
        plt.savefig(auroc_path)
        plt.close()
        
        return auroc_path


def plot_auprc(y_true, y_pred_proba, classifier_name, output_dir, auprc_score=None):
    """
    Plot AUPRC curve for binary or multiclass classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classifier_name: Name of the classifier
        output_dir: Directory to save the plot
        auprc_score: Pre-calculated AUPRC score (optional, will be calculated if None)
    
    Returns:
        str or tuple: Path to saved AUPRC plot (and calculated AUPRC score for multiclass)
    """
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2
    
    # binary classification
    if is_binary:
        plt.figure(figsize=(8, 6))
        
        # Handle different probability formats for binary classification
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            # If shape is (n_samples, 2), use positive class probabilities
            y_pred_proba_positive = y_pred_proba[:, 1]
        elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
            # If shape is (n_samples, 1), flatten it
            y_pred_proba_positive = y_pred_proba.ravel()
        else:
            # If shape is (n_samples,), use as is
            y_pred_proba_positive = y_pred_proba
        
        # Calculate precision-recall curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba_positive)
        
        plt.plot(recall_vals, precision_vals, color='blue', label=f'AUPRC = {auprc_score:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'AUPRC - {classifier_name}')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.tight_layout()
        
        auprc_path = os.path.join(output_dir, 'AUPRC.jpg')
        plt.savefig(auprc_path)
        plt.close()
        
        return auprc_path
    
    # Multiclass classification
    else:
        classes = unique_classes
        
        # Binarize the output
        y_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_bin = np.hstack((1 - y_bin, y_bin))
        
        # Compute PR curve and AUPRC for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
            average_precision[i] = average_precision_score(y_bin[:, i], y_pred_proba[:, i])
        
        # Calculate overall AUPRC (weighted by class support)
        auprc_ovr = average_precision_score(y_bin, y_pred_proba, average='weighted')
        
        # Plot all PR curves
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'Class {classes[i]} (AUPRC = {average_precision[i]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Multiclass Precision-Recall Curves (One-vs-Rest) - {classifier_name}\nOverall AUPRC = {auprc_ovr:.3f}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        
        auprc_path = os.path.join(output_dir, 'AUPRC.jpg')
        plt.savefig(auprc_path)
        plt.close()
        
        return auprc_path, auprc_ovr