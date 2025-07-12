import os
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import shap
from ..tools import (
    plot_auroc, plot_auprc, plot_confusion_matrix
)


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers.
    Implements common functionality and defines the interface to be implemented in derived classes.
    """
    
    def __init__(self, name, logger=None):
        """
        Initialize the classifier.
        
        Args:
            name (str): Name of the classifier
            logger: Logger instance for tracking progress
        """
        self.name = name
        self.model = None
        self.best_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.logger = logger
        
        self.log('debug', f"Initialized {name} classifier")

    @abstractmethod
    def train(self, X, y) -> None:
        """
        Train the specific model

        Args:
            X: Training data
            y: Training labels
        """
        pass

    @abstractmethod
    def _predict_proba(self, X):
        """
        Specific implementation of probability prediction.
        
        Args:
            X: Preprocessed data
            
        Returns:
            Predicted probabilities
        """
        pass
    

    @abstractmethod
    def create_estimator(self, params=None):
        """
        Creates a model instance with the specified parameters.
        
        Args:
            params: Parameters for the model
            
        Returns:
            Model instance
        """
        pass


    @abstractmethod
    def _get_shap_values(self, X_scaled):
        """
        Calculate SHAP values for the specific model.
        
        Args:
            X_scaled: Standardized data
            
        Returns:
            List of arrays: SHAP values for each class.
            For binary classification, returns [shap_class_0, shap_class_1].
            For multiclass classification, returns [shap_class_0, shap_class_1, ..., shap_class_n].
            Each array has shape (n_samples, n_features).
        """
        pass


    def log(self, level, message):
        """
        Log a message if logger is available.
        
        Args:
            level: Log level (info, debug, warning, error, critical)
            message: Message to log
        """
        if not self.logger:
            # Fall back to print if no logger is provided
            if level in ['info', 'warning', 'error', 'critical']:
                print(f"[{self.name.upper()} {level.upper()}] {message}")
            return
            
        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
    

    def preprocess_data(self, X, y=None):
        """
        Preprocesses data by applying standardization.
        
        Args:
            X: Data to preprocess
            y: Labels (optional)
            
        Returns:
            Preprocessed X (and y if provided)
        """
        self.log('debug', f"Preprocessing data with shape {X.shape}")
        if y is not None:
            return self.scaler.transform(X), y
        return self.scaler.transform(X)
    

    def fit_scaler(self, X):
        """
        Trains the scaler on training data.
        
        Args:
            X: Training data
        """
        self.log('debug', f"Fitting scaler to data with shape {X.shape}")
        self.scaler.fit(X)
    

    def grid_search(self, X, y, param_grid):
        """
        Performs a grid search to find the best hyperparameters.
        
        Args:
            X: Training data
            y: Training labels
            param_grid: Grid of parameters to test
                
        Returns:
            The model trained with the best parameters
        """
        self.log('info', f"Starting GridSearchCV for {self.name} with {len(param_grid)} configurations")
        
        if self.logger:
            self.logger.start_timer(f"grid_search_{self.name}")
        
        # Create a base estimator
        base_estimator = self.create_estimator()
        
        # Determine whether to use subsampling based on data size
        should_subsample = True if len(X) > 50000 else False
        
        # Configure the cross-validation method
        if should_subsample:
            self.log('info', f"Large dataset detected ({len(X)} samples)")
            self.log('info', f"Using subsampling with 3 folds of max size 50000")
            
            from sklearn.model_selection import StratifiedShuffleSplit
            cv = StratifiedShuffleSplit(
                n_splits=3,
                test_size=50000,
                random_state=42,
            )
        else:
            # Standard CV
            cv = 2
            self.log('debug', f"Using standard cross-validation with {cv} folds")
        
        # Determine if this is binary or multiclass classification
        unique_classes = np.unique(y)
        scoring = 'f1' if len(unique_classes) <= 2 else 'f1_weighted'
        self.log('debug', f"Detected {len(unique_classes)} classes, using scoring metric: {scoring}")
        
        # Configure and run GridSearchCV
        grid = GridSearchCV(
            base_estimator, 
            param_grid,
            scoring=scoring,  # Use appropriate scoring based on problem type
            cv=cv,  # Use the defined CV method
            n_jobs=-1, 
            verbose=0  # Reduced to avoid duplicate output with logger
        )
        
        self.log('info', f"Fitting GridSearchCV with {len(X)} samples, {len(param_grid)} parameter combinations")
        
        # Run the grid search
        grid.fit(X, y)
        
        # Save the best parameters
        self.best_params = grid.best_params_
        self.best_model = grid.best_estimator_
        
        self.model = self.best_model
        self.is_fitted = True
        
        self.log('info', f"Best parameters found: {self.best_params}")
        self.log('info', f"Best CV score: {grid.best_score_:.4f}")
        
        # Log performance of all tested parameters
        if self.logger:
            self.logger.stop_timer(f"grid_search_{self.name}")
            
            # Create a dictionary with formatted results
            results_dict = {}
            for i, params in enumerate(grid.cv_results_['params']):
                param_key = ", ".join([f"{k}={v}" for k, v in params.items()])
                mean_score = grid.cv_results_['mean_test_score'][i]
                std_score = grid.cv_results_['std_test_score'][i]
                results_dict[param_key] = f"{mean_score:.4f} (±{std_score:.4f})"
            
            self.logger.log_result(f"{self.name} Parameter Search Results", results_dict)
        
        return self.best_model, self.best_params, grid.best_score_
    

    def predict(self, X):
        """
        Makes predictions on the provided data.
        
        Args:
            X: Data for predictions
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            error_msg = "Model must be trained before making predictions"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        self.log('debug', f"Predicting on {X.shape[0]} samples")
        X_scaled = self.preprocess_data(X)
        return self._predict_proba(X_scaled)
    

    def _log_metrics(self, metrics):
        self.log('info', f"Evaluation results for {self.name}:")
        self.log('info', f"  - Accuracy: {metrics['accuracy']:.4f}")
        self.log('info', f"  - Precision: {metrics['precision']:.4f}")
        self.log('info', f"  - Recall: {metrics['recall']:.4f}")
        self.log('info', f"  - F1-score: {metrics['f1_score']:.4f}")
        self.log('info', f"  - AUROC: {metrics['auroc']:.4f}")
        self.log('info', f"  - AUPRC: {metrics['auprc']:.4f}")
        self.log('info', f"  - Specificity: {metrics['specificity']:.4f}")
        self.log('debug', f"Confusion Matrix: [[{metrics['true_negative']}, {metrics['false_positive']}], [{metrics['false_negative']}, {metrics['true_positive']}]]")


    def _save_metrics(self, metrics, output_dir, save_pickle: bool = True, save_csv: bool = True):
        if save_pickle:
            metrics_df = pd.DataFrame([metrics])
            metrics_pkl_path = os.path.join(output_dir, 'metrics.pkl')
            metrics_df.to_pickle(metrics_pkl_path)
            self.log('debug', f"Saved metrics pickle to {metrics_pkl_path}")
        if save_csv:
            metrics_simple = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
            metrics_csv_path = os.path.join(output_dir, 'metrics.csv')
            pd.DataFrame([metrics_simple]).to_csv(metrics_csv_path, index=False)
            self.log('debug', f"Saved metrics CSV to {metrics_csv_path}")


    def evaluate(self, X, y, output_dir=None):
        """
        Evaluates the model on test data and automatically determines if it's a binary
        or multiclass classification task.
        
        Args:
            X: Test data
            y: Test labels
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            error_msg = "Model must be trained before evaluation"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        # Determine task type (binary or multiclass classification) based on number of unique classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        self.log('info', f"Detected {n_classes} unique classes in labels")
        
        if n_classes <= 2:
            # Binary classification
            self.log('info', f"Using binary classification evaluation for {self.name}")
            return self._evaluate_binary(X, y, output_dir)
        else:
            # Multiclass classification
            self.log('info', f"Using multiclass classification evaluation for {self.name}")
            return self._evaluate_multiclass(X, y, output_dir)


    def _evaluate_binary(self, X, y, output_dir=None):
        """
        Evaluates the model on binary classification test data.
        
        Args:
            X: Test data
            y: Test labels
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.log('info', f"Evaluating {self.name} on {X.shape[0]} samples (binary classification)")
        
        if self.logger:
            self.logger.start_timer(f"evaluate_{self.name}")
        
        # Preprocessing and prediction
        X_scaled = self.preprocess_data(X)
        y_pred_proba = self._predict_proba(X_scaled)
        
        # Handle different probability output formats
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            # If predict_proba returns shape (n_samples, 2), use positive class probabilities
            y_pred_proba_positive = y_pred_proba[:, 1]
        elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
            # If predict_proba returns shape (n_samples, 1), flatten it
            y_pred_proba_positive = y_pred_proba.ravel()
        else:
            # If predict_proba returns shape (n_samples,), use as is
            y_pred_proba_positive = y_pred_proba
        
        y_pred = (y_pred_proba_positive >= 0.5).astype(int)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auroc = roc_auc_score(y, y_pred_proba_positive)
        auprc = average_precision_score(y, y_pred_proba_positive)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Data for ROC and PR curves
        fpr, tpr, _ = roc_curve(y, y_pred_proba_positive)
        precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba_positive)
        
        # Collect metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auroc': auroc,
            'auprc': auprc,
            'specificity': specificity,
            'npv': npv,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'confusion_matrix': cm,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'precision_vals': precision_vals,
            'recall_vals': recall_vals
        }
        
        # Log metrics
        self._log_metrics(metrics)
        
        # Create and save plots and metrics summary if directory provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log('debug', f"Created output directory: {output_dir}")
            
            plot_auroc(y, y_pred_proba_positive, self.name, output_dir, auroc)
            plot_auprc(y, y_pred_proba_positive, self.name, output_dir, auprc)
            plot_confusion_matrix(cm, self.name, output_dir)
            self._save_metrics(metrics, output_dir)
        
        if self.logger:
            self.logger.stop_timer(f"evaluate_{self.name}")
            
            # Log detailed metrics
            self.logger.log_result(f"{self.name} Evaluation Metrics", {
                "accuracy": f"{accuracy:.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1_score": f"{f1:.4f}",
                "AUROC": f"{auroc:.4f}",
                "AUPRC": f"{auprc:.4f}",
                "specificity": f"{specificity:.4f}",
                "confusion matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]"
            })
        
        return metrics


    def _evaluate_multiclass(self, X, y, output_dir=None):
        """
        Evaluates the model on multiclass classification test data.
        
        Args:
            X: Test data
            y: Test labels
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.log('info', f"Evaluating {self.name} on {X.shape[0]} samples (multiclass classification)")
        
        if self.logger:
            self.logger.start_timer(f"evaluate_multiclass_{self.name}")
        
        # Preprocessing and prediction
        X_scaled = self.preprocess_data(X)
        y_pred_proba = self._predict_proba(X_scaled)
        
        # For multiclass, predict_proba returns an array of shape (n_samples, n_classes)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get list of unique class labels
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y, y_pred)
        
        # use weighted scores for multiclass metrics
        precision_weighted = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # One-vs-rest ROC AUC (each class against all others)
        auroc_ovr = roc_auc_score(
            y, 
            y_pred_proba,
            average='weighted',
            multi_class='ovr'
        )
        
        # Calculate AUPRC for multiclass
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y, classes=classes)
        if n_classes == 2:
            y_bin = np.hstack((1 - y_bin, y_bin))
        auprc_ovr = average_precision_score(y_bin, y_pred_proba, average='weighted')
        
        # Collect metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1_score': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'auroc_ovr': auroc_ovr,
            'auprc_ovr': auprc_ovr,
            'confusion_matrix': cm,
            'classes': classes
        }
        
        # Log metrics
        self._log_multiclass_metrics(metrics)
        
        # Create and save plots and metrics summary if directory provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log('debug', f"Created output directory: {output_dir}")
            
            plot_confusion_matrix(cm, self.name, output_dir, classes)
            plot_auroc(y, y_pred_proba, self.name, output_dir, auroc_ovr)
            auprc_result = plot_auprc(y, y_pred_proba, self.name, output_dir)
            if isinstance(auprc_result, tuple):
                auprc_path, auprc_ovr_calc = auprc_result
            else:
                auprc_path = auprc_result
            self._save_metrics(metrics, output_dir)
        
        if self.logger:
            self.logger.stop_timer(f"evaluate_multiclass_{self.name}")
            
            # Log detailed metrics
            log_metrics = {
                "Accuracy": f"{accuracy:.4f}",
                "Precision (weighted)": f"{precision_weighted:.4f}",
                "Recall (weighted)": f"{recall_weighted:.4f}",
                "F1-score (weighted)": f"{f1_weighted:.4f}",
                "AUROC (one-vs-rest)": f"{auroc_ovr:.4f}",
                "AUPRC (one-vs-rest)": f"{auprc_ovr:.4f}"
            }
            
            # Add per-class metrics
            for i, cls in enumerate(classes):
                log_metrics[f"Class {cls} Precision"] = f"{precision_per_class[i]:.4f}"
                log_metrics[f"Class {cls} Recall"] = f"{recall_per_class[i]:.4f}"
                log_metrics[f"Class {cls} F1"] = f"{f1_per_class[i]:.4f}"
            
            self.logger.log_result(f"{self.name} Multiclass Evaluation Metrics", log_metrics)
        
        return metrics


    def _log_multiclass_metrics(self, metrics):
        """
        Log multiclass metrics to the logger.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        self.log('info', f"Multiclass evaluation results for {self.name}:")
        self.log('info', f"  - Accuracy: {metrics['accuracy']:.4f}")
        self.log('info', f"  - Weighted Precision: {metrics['precision']:.4f}")
        self.log('info', f"  - Weighted Recall: {metrics['recall']:.4f}")
        self.log('info', f"  - Weighted F1-score: {metrics['f1_score']:.4f}")
        self.log('info', f"  - OvR AUROC: {metrics['auroc_ovr']:.4f}")
        self.log('info', f"  - OvR AUPRC: {metrics['auprc_ovr']:.4f}")
        
        # Log per-class metrics
        for i, cls in enumerate(metrics['classes']):
            self.log('debug', f"  Class {cls}:")
            self.log('debug', f"    - Precision: {metrics['precision_per_class'][i]:.4f}")
            self.log('debug', f"    - Recall: {metrics['recall_per_class'][i]:.4f}")
            self.log('debug', f"    - F1-score: {metrics['f1_per_class'][i]:.4f}")


    def load_model(self, input_path):
        """
        Loads a model from disk.
        
        Args:
            input_path: Path of the saved model
        """
        self.log('info', f"Loading model from {input_path}")
        
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
                
            self.best_model = data['model']
            self.best_params = data['params']
            self.scaler = data['scaler']
            self.model = self.best_model
            self.is_fitted = True
            
            # Update name if present in the saved model
            if 'name' in data:
                loaded_name = data['name']
                if loaded_name != self.name:
                    self.log('warning', f"Loaded model name '{loaded_name}' differs from current model name '{self.name}'")
            
            self.log('info', f"Model loaded successfully with parameters: {self.best_params}")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)

    
    def save_model(self, output_path):
        """
        Saves the model to disk.
        
        Args:
            output_path: Path where to save the model
        """

        if not self.is_fitted:
            error_msg = "Model must be trained before saving"
            self.log('error', error_msg)
            raise ValueError(error_msg)
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.log('info', f"Saving {self.name} model to {output_path}")
        
        # Save the model
        try:
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'params': self.best_params,
                    'scaler': self.scaler,
                    'name': self.name
                }, f)
            self.log('info', f"Model saved successfully")
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
  

    def calculate_SHAP_values(self, X, output_path):
        """
        Calculates SHAP values for each feature and saves summary plots.
        Supports both binary and multiclass classification.

        Args:
            X (pd.DataFrame or np.ndarray): Input data to explain.
            output_path (str): Directory where to save SHAP plots.

        Returns:
            pd.DataFrame: 
                Columns:
                - 'feature': feature names
                - 'shap_class_<i>': mean |SHAP| for class i (for each class)
                - 'shap_total': sum of all class SHAP values
        """
        # Verify the model has been trained
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            raise RuntimeError("The model must be trained before calculating SHAP values.")

        # Prepare DataFrame
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            feature_names = X_df.columns.tolist()
        else:
            X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
            feature_names = X_df.columns.tolist()

        # Standardize features
        X_scaled = self.scaler.transform(X)

        # Calculate SHAP values using the explainer specific to the model type
        shap_values_list = self._get_shap_values(X_scaled)

        # Handle both old binary format (tuple) and new list format
        if isinstance(shap_values_list, tuple):
            # Convert tuple to list for backward compatibility
            shap_values_list = list(shap_values_list)
        elif not isinstance(shap_values_list, list):
            # Single array case - wrap in list
            shap_values_list = [shap_values_list]

        n_classes = len(shap_values_list)
        self.log('info', f"Calculating SHAP values for {n_classes} classes")

        # Debug: Log shapes to understand the structure
        for i, shap_vals in enumerate(shap_values_list):
            self.log('debug', f"shap_class_{i} shape: {shap_vals.shape}")
        self.log('debug', f"feature_names length: {len(feature_names)}")

        # Ensure all SHAP values have the correct shape (n_samples, n_features)
        for i, shap_vals in enumerate(shap_values_list):
            if shap_vals.ndim == 1:
                # If 1D, it should be reshaped to (1, n_features) or (n_features,)
                if len(shap_vals) == len(feature_names):
                    # Single sample case - reshape to (1, n_features)
                    shap_values_list[i] = shap_vals.reshape(1, -1)
                elif len(shap_vals) == X_scaled.shape[0]:
                    # This means we have (n_samples,) instead of (n_samples, n_features)
                    # This is an error - SHAP should return (n_samples, n_features)
                    raise ValueError(f"SHAP values for class {i} have unexpected shape. Expected (n_samples, n_features), got ({shap_vals.shape[0]},)")

            # Verify shapes match
            if shap_values_list[i].shape[1] != len(feature_names):
                raise ValueError(f"SHAP values shape mismatch for class {i}. SHAP shape: {shap_values_list[i].shape}, features: {len(feature_names)}")

        # Calculate mean absolute SHAP per feature for each class
        class_means = []
        df_data = {'feature': feature_names}
        
        for i, shap_vals in enumerate(shap_values_list):
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            class_means.append(mean_abs_shap)
            df_data[f'shap_class_{i}'] = mean_abs_shap

        # Calculate total SHAP importance across all classes
        total_shap = np.sum(class_means, axis=0)
        df_data['shap_total'] = total_shap

        # Build result DataFrame
        df_data = pd.DataFrame(df_data)

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Save summary plots for each class
        for i, shap_vals in enumerate(shap_values_list):
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_scaled, feature_names=feature_names, max_display=20, show=False)
            plt.title(f"SHAP summary – class {i}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"shap_summary_class{i}.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # Create a combined plot showing total SHAP importance
        plt.figure(figsize=(10, 6))
        # Sort features by total importance
        sorted_indices = np.argsort(total_shap)[::-1][:20]  # Top 20 features
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_totals = total_shap[sorted_indices]
        
        # Create horizontal bar plot
        plt.barh(range(len(sorted_features)), sorted_totals)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Total SHAP Importance (sum across all classes)')
        plt.title('Feature Importance (Total SHAP values)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "shap_total_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save DataFrame as CSV
        df_data.to_csv(os.path.join(output_path, "shap_importances.csv"), index=False)

        self.log('info', f"SHAP analysis completed for {n_classes} classes. Results saved to {output_path}")

        return df_data
