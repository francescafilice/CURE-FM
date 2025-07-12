import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .Classifiers.ClassifierFactory import ClassifierFactory
import umap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .tools import plot_umap_projection, get_balanced_samples


class ClassifiersHandler:
    """
    Class for training, evaluating, and selecting hyperparameters of classifiers
    using the folds created by DataSampler.
    """
    
    def __init__(self, sampler, classification_params, results_dir='results', logger=None):
        """
        Initialize the classifier.
        
        Args:
            sampler: The data sampler (DataSampler)
            classification_params: Configuration parameters for classification
            results_dir: Directory where to save the results
            logger: Logger instance for tracking progress
        """
        self.sampler = sampler
        self.classification_params = classification_params
        self.results_dir = results_dir
        self.best_models = {}
        self.logger = logger
        
        # Create main results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.log('info', f"Classifier initialized with results directory: {self.results_dir}")
        self.log('debug', f"Classification parameters: {classification_params}")
    
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
                print(f"[{level.upper()}] {message}")
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
    
    def _get_classifier_config(self, classifier_type):
        """
        Get the configuration for a specific classifier.
        
        Args:
            classifier_type: Type of classifier
            
        Returns:
            The classifier configuration or None if not found
        """
        for classifier_config in self.classification_params.get('classifiers', []):
            if classifier_config.get('type') == classifier_type:
                return classifier_config
        
        self.log('warning', f"Configuration not found for classifier type: {classifier_type}")
        return None
    
    def _prepare_output_directory(self, pool_method, target_label, classifier_type):
        """
        Prepare the output directory for a specific classifier.
        
        Args:
            pool_method: Pooling method
            target_label: Target label
            classifier_type: Type of classifier
            
        Returns:
            Path to the output directory
        """
        output_dir = os.path.join(
            self.results_dir, 
            pool_method, 
            target_label, 
            classifier_type
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.log('debug', f"Created output directory: {output_dir}")
        
        return output_dir
    
    def train_classifiers(self, pool_method, target_label):
        """
        Train all enabled classifiers for a specific target label and pooling method,
        using cross-validation for hyperparameter selection.
        
        Args:
            pool_method: Pooling method
            target_label: Target label
            
        Returns:
            Dictionary with results for each classifier
        """
        start_time = time.time()
        self.log('info', f"Training classifiers for {target_label} using {pool_method} pooling method...")
        
        # Get folds for cross-validation
        fold_indices = self.sampler.create_kfold_indices(pool_method, target_label)
        n_folds = len(fold_indices)
        self.log('info', f"Using {n_folds}-fold cross-validation")
        
        results = {}
        
        # For each classifier configuration in the config
        for classifier_config in self.classification_params.get('classifiers', []):
            classifier_type = classifier_config.get('type')
            
            # Skip disabled classifiers
            if not classifier_config.get('enabled', True):
                self.log('info', f"Classifier {classifier_type} is disabled, skipping.")
                continue
            
            self.log('info', f"Training {classifier_type} classifier")
            
            if self.logger:
                self.logger.start_timer(f"train_{classifier_type}_{pool_method}_{target_label}")
            
            # Prepare output directory
            output_dir = self._prepare_output_directory(pool_method, target_label, classifier_type)
            
            # Results for each fold
            fold_results = []
            best_parameters = []
            best_metric_values = []
            best_models = []
            
            # Log classifier hyperparameters
            hyperparams = classifier_config.get('hyperparameters', {})
            self.log('debug', f"Hyperparameter search space for {classifier_type}:")
            for param, values in hyperparams.items():
                self.log('debug', f"  - {param}: {values}")
            
            # For each fold, train and evaluate the classifier
            for fold_idx in range(n_folds):
                self.log('info', f"Processing fold {fold_idx+1}/{n_folds}")
                
                if self.logger:
                    self.logger.start_timer(f"fold_{fold_idx}_{classifier_type}")
                
                # Get fold data
                X_train, X_val, y_train, y_val = self.sampler.get_fold_data(
                    pool_method, target_label, fold_idx)
                
                self.log('debug', f"Fold {fold_idx+1} data: {X_train.shape[0]} training samples, "
                               f"{X_val.shape[0]} validation samples, {X_train.shape[1]} features")
                
                # Create a new classifier instance
                classifier = ClassifierFactory.create_classifier(classifier_type, logger=self.logger)
                
                # Directory for this fold
                fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)
                
                # Create parameter grid
                param_grid = {}
                for key, values in hyperparams.items():
                    if isinstance(values, list):
                        param_grid[key] = values
                    else:
                        # For single parameters (like epochs, val_split)
                        param_grid[key] = values
                
                # Train the classifier
                self.log('info', f"Starting training for fold {fold_idx+1}")
                training_results = classifier.train(X_train, y_train, param_grid=param_grid)
                
                # Log best parameters for this fold
                self.log('info', f"Best parameters for fold {fold_idx+1}: {training_results['best_params']}")
                
                # Evaluate the classifier
                self.log('info', f"Evaluating fold {fold_idx+1}")
                metrics = classifier.evaluate(X_val, y_val, output_dir=fold_dir)
                
                # Log fold metrics
                self.log('info', f"Fold {fold_idx+1} metrics:")
                for metric_name, metric_value in metrics.items():
                    if not isinstance(metric_value, np.ndarray):
                        self.log('info', f"  - {metric_name}: {metric_value:.6f}")
                
                # Save the model and results
                classifier.save_model(os.path.join(fold_dir, f'model.pkl'))
                self._save_intermediate_results(fold_dir, training_results['best_params'], metrics)

                
                fold_results.append(metrics)
                best_parameters.append(training_results['best_params'])
                best_metric_values.append(metrics['f1_score'])
                best_models.append(classifier)
                
                # Log primary metric progress
                self.log('info', f"Fold {fold_idx+1} completed - f1_score: {metrics['f1_score']:.4f}")
                
                if self.logger:
                    self.logger.stop_timer(f"fold_{fold_idx}_{classifier_type}")
            
            # Select the best model across all folds
            best_fold_idx = np.argmax(best_metric_values)
            best_fold_model = best_models[best_fold_idx]
            best_fold_params = best_parameters[best_fold_idx]
            best_fold_metrics = fold_results[best_fold_idx]
            
            self.log('info', f"Best model from fold {best_fold_idx+1} - "
                          f"f1_score: {best_fold_metrics['f1_score']:.4f}")
            
            # Save the best model in a separate folder
            best_dir = os.path.join(output_dir, 'best_model')
            if not os.path.exists(best_dir):
                os.makedirs(best_dir)
            
            best_fold_model.save_model(os.path.join(best_dir, 'model.pkl'))
            self._save_intermediate_results(best_dir, best_fold_params, best_fold_metrics)
            
            # Check if we need to extract SHAP values
            classifier_config = self._get_classifier_config(classifier_type)
            extract_shap = classifier_config.get('extract_shap', False)
            extract_umap = classifier_config.get('extract_umap', False)
            
            if extract_shap:
                # Calculate SHAP values using the dedicated function
                self.log('info', f"Extract SHAP flag is set for {classifier_type}. Calculating SHAP values...")
                shap_dir = os.path.join(best_dir, 'shap')
                shap_results = self.calculate_model_shap_umap(best_fold_model, 
                                                              best_fold_idx, 
                                                              pool_method, 
                                                              target_label, 
                                                              output_dir=shap_dir,
                                                              extract_umap=extract_umap)
            
            # Add the best model to the dictionary of best models
            self._add_best_model(target_label, pool_method, classifier_type, 
                                best_fold_model, best_fold_params, best_fold_metrics)
            
            # Add results to the results dictionary
            self._add_results(results, target_label, pool_method, classifier_type, 
                             fold_results, best_fold_idx, best_fold_params, best_fold_metrics)
            
            # Calculate and save median metrics
            median_metrics = self._calculate_median_metrics(fold_results)
            summary_df = pd.DataFrame([median_metrics])
            summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
            
            # Log model performance summary
            if self.logger:
                self.logger.log_model_summary(
                    model_name=f"{classifier_type} ({pool_method}, {target_label})",
                    model_params=best_fold_params,
                    metrics={**best_fold_metrics, 
                             'avg_accuracy': median_metrics['accuracy'],
                             'avg_f1': median_metrics.get('f1', 'N/A')}
                )
            
            self.log('info', f"Completed training {classifier_type} - Best f1_score: {best_fold_metrics['f1_score']:.4f}")
            
            if self.logger:
                self.logger.stop_timer(f"train_{classifier_type}_{pool_method}_{target_label}")
        
        elapsed_time = time.time() - start_time
        self.log('info', f"Training completed in {elapsed_time:.2f} seconds")
        
        # Log memory usage after training
        if self.logger:
            self.logger.log_memory_usage()
            
        return results
    
    def _add_best_model(self, target_label, pool_method, classifier_type, model, params, metrics):
        """
        Add a best model to the dictionary of best models.
        
        Args:
            target_label: Target label
            pool_method: Pooling method
            classifier_type: Type of classifier
            model: Best model
            params: Best parameters
            metrics: Best metrics
        """
        if target_label not in self.best_models:
            self.best_models[target_label] = {}
        if pool_method not in self.best_models[target_label]:
            self.best_models[target_label][pool_method] = {}
        
        self.best_models[target_label][pool_method][classifier_type] = {
            'model': model,
            'parameters': params,
            'metrics': metrics
        }
        
        self.log('debug', f"Added best model for {target_label}, {pool_method}, {classifier_type} to registry")
    
    def _add_results(self, results, target_label, pool_method, classifier_type, 
                    fold_results, best_fold_idx, best_params, best_metrics):
        """
        Add results to the results dictionary.
        
        Args:
            results: Results dictionary
            target_label: Target label
            pool_method: Pooling method
            classifier_type: Type of classifier
            fold_results: Results for each fold
            best_fold_idx: Index of the best fold
            best_params: Best parameters
            best_metrics: Best metrics
        """
        if target_label not in results:
            results[target_label] = {}
        if pool_method not in results[target_label]:
            results[target_label][pool_method] = {}
        
        # Calculate median metrics across all folds
        median_metrics = self._calculate_median_metrics(fold_results)
        
        results[target_label][pool_method][classifier_type] = {
            'median_metrics': median_metrics,
            'best_fold': {
                'fold_idx': best_fold_idx,
                'parameters': best_params,
                'metrics': best_metrics
            }
        }
        
        # Log detailed metrics
        self.log('debug', f"Added results for {target_label}, {pool_method}, {classifier_type}")
        self.log('debug', f"Median metrics: {median_metrics}")
    
    def _calculate_median_metrics(self, metrics_list):
        """
        Calculate the median of metrics across multiple folds.
        
        Args:
            metrics_list: List of dictionaries with metrics
            
        Returns:
            Dictionary with median metrics
        """
        if not metrics_list:
            return {}
            
        # Get metric keys from the first dictionary (excluding numpy arrays)
        metric_keys = [k for k in metrics_list[0].keys() if not isinstance(metrics_list[0][k], np.ndarray)]
        
        # Initialize dictionary for median metrics
        median_metrics = {}
        
        # Calculate the median for each metric
        for key in metric_keys:
            # Extract values for this metric across all folds
            values = [metrics[key] for metrics in metrics_list]
            # Calculate median
            median_metrics[key] = np.median(values)
        
        return median_metrics
    
    def train_all(self):
        """
        Train all classifiers for all target labels and all pooling methods.
        
        Returns:
            Dictionary with all results
        """
        self.log('info', "Starting training for all classifiers, target labels, and pooling methods")
        
        if self.logger:
            self.logger.log_step("Classification Training", "Training and evaluating all classifiers")
            self.logger.start_timer("train_all")
        
        all_results = {}
        target_labels = self.sampler.get_all_target_labels()
        pooling_methods = self.sampler.get_all_pooling_methods()
        
        self.log('info', f"Target labels to process: {target_labels}")
        self.log('info', f"Pooling methods to process: {pooling_methods}")
        
        for target_label in target_labels:
            self.log('info', f"Processing target label: {target_label}")
            
            if self.logger:
                self.logger.log_step(f"Target Label: {target_label}", "Training for all pooling methods")
                self.logger.start_timer(f"train_label_{target_label}")
            
            for pool_method in pooling_methods:
                self.log('info', f"Processing pooling method: {pool_method}")
                
                if self.logger:
                    self.logger.start_timer(f"train_{pool_method}_{target_label}")
                
                results = self.train_classifiers(pool_method, target_label)
                
                if target_label not in all_results:
                    all_results[target_label] = {}
                
                all_results[target_label][pool_method] = results.get(target_label, {}).get(pool_method, {})
                
                if self.logger:
                    self.logger.stop_timer(f"train_{pool_method}_{target_label}")
            
            if self.logger:
                self.logger.stop_timer(f"train_label_{target_label}")
        
        # Save a general summary
        self._save_summary(all_results)
        
        if self.logger:
            self.logger.stop_timer("train_all")
            self.logger.log_memory_usage()
        
        self.log('info', "Training completed for all classifiers, target labels, and pooling methods")
        
        return all_results
    
    def _save_summary(self, all_results):
        """
        Save a general summary of results.
        
        Args:
            all_results: Dictionary with all results
        """
        self.log('info', "Generating and saving summary of all results")
        
        summary_rows = []
        
        for target_label in all_results:
            for pool_method in all_results[target_label]:
                for classifier_type in all_results[target_label][pool_method]:
                    median_metrics = all_results[target_label][pool_method][classifier_type]['median_metrics']
                    best_metrics = all_results[target_label][pool_method][classifier_type]['best_fold']['metrics']
                    
                    row = {
                        'target_label': target_label,
                        'pool_method': pool_method,
                        'classifier': classifier_type,
                        **{f'avg_{k}': v for k, v in median_metrics.items() if not isinstance(v, np.ndarray)},
                        **{f'best_{k}': v for k, v in best_metrics.items() if not isinstance(v, np.ndarray)}
                    }
                    
                    summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(self.results_dir, 'all_results_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            self.log('info', f"Summary saved in {summary_path}")
            
            # Create a more readable summary for logging
            if self.logger:
                # Group by target label
                summary_by_label = {}
                for row in summary_rows:
                    label = row['target_label']
                    if label not in summary_by_label:
                        summary_by_label[label] = []
                    summary_by_label[label].append(row)
                
                # Log summary for each label
                for label, rows in summary_by_label.items():
                    # Sort by accuracy (best model first)
                    sorted_rows = sorted(rows, key=lambda x: x.get('best_f1', 0), reverse=True)
                    top_model = sorted_rows[0]
                    
                    # Log the best model for this label
                    self.logger.log_result(
                        f"Best Model for {label}",
                        {
                            "Classifier": top_model['classifier'],
                            "Pooling Method": top_model['pool_method'],
                            "Accuracy": f"{top_model.get('best_accuracy', 0):.4f}",
                            "F1 Score": f"{top_model.get('best_f1', 0):.4f}" if 'best_f1' in top_model else "N/A",
                            "Precision": f"{top_model.get('best_precision', 0):.4f}" if 'best_precision' in top_model else "N/A",
                            "Recall": f"{top_model.get('best_recall', 0):.4f}" if 'best_recall' in top_model else "N/A"
                        }
                    )
    
    def get_best_model(self, target_label, pool_method, classifier_type=None):
        """
        Get the best model for a specific target label, pooling method, and classifier type.
        If classifier_type is None, return the best model across all classifier types.
        
        Args:
            target_label: Target label
            pool_method: Pooling method
            classifier_type: Type of classifier (optional)
            
        Returns:
            Best model information or None if not found
        """
        if target_label not in self.best_models:
            self.log('warning', f"No models found for target label: {target_label}")
            return None
        if pool_method not in self.best_models[target_label]:
            self.log('warning', f"No models found for pooling method: {pool_method}")
            return None
            
        if classifier_type:
            # Return specific classifier if requested
            if classifier_type not in self.best_models[target_label][pool_method]:
                self.log('warning', f"Classifier type not found: {classifier_type}")
                return None
                
            model_info = self.best_models[target_label][pool_method].get(classifier_type)
            self.log('info', f"Retrieved specific model: {classifier_type} for {target_label}, {pool_method}")
            return model_info
        else:
            # Return the best classifier across all types based on accuracy
            best_acc = -1
            best_clf = None
            best_type = None
            
            for clf_type, clf_data in self.best_models[target_label][pool_method].items():
                if clf_data['metrics']['f1_score'] > best_acc:
                    best_acc = clf_data['metrics']['f1_score']
                    best_clf = clf_data
                    best_type = clf_type
            
            self.log('info', f"Retrieved best model: {best_type} for {target_label}, {pool_method} "
                         f"(f1_score: {best_acc:.4f})")
            return best_clf
    
    def log_hyperparameter_analysis(self, target_label=None, pool_method=None, classifier_type=None):
        """
        Analyze and log the impact of hyperparameters on model performance.
        
        Args:
            target_label: Optional specific target label to analyze
            pool_method: Optional specific pooling method to analyze
            classifier_type: Optional specific classifier type to analyze
        """
        if self.logger is None:
            self.log('warning', "Logger not available, skipping hyperparameter analysis")
            return
        
        self.log('info', "Performing hyperparameter analysis")
        
        target_labels = [target_label] if target_label else self.best_models.keys()
        
        for label in target_labels:
            if label not in self.best_models:
                continue
                
            pool_methods = [pool_method] if pool_method else self.best_models[label].keys()
            
            for method in pool_methods:
                if method not in self.best_models[label]:
                    continue
                    
                classifier_types = [classifier_type] if classifier_type else self.best_models[label][method].keys()
                
                for clf_type in classifier_types:
                    if clf_type not in self.best_models[label][method]:
                        continue
                        
                    model_info = self.best_models[label][method][clf_type]
                    params = model_info['parameters']
                    metrics = model_info['metrics']
                    
                    self.logger.log_result(
                        f"Hyperparameter Analysis: {clf_type} ({method}, {label})",
                        {
                            **{f"param_{k}": v for k, v in params.items()},
                            **{f"metric_{k}": f"{v:.6f}" if isinstance(v, float) else v 
                               for k, v in metrics.items() if not isinstance(v, np.ndarray)}
                        }
                    )
                    
        self.log('info', "Hyperparameter analysis completed")
        
    def _save_intermediate_results(self, directory, params, metrics, prefix=''):
        """
        Saves the parameters and metrics in pickle format and in a single readable CSV file.
        
        Args:
            directory: Directory where to save the files
            params: Dictionary with parameters to save
            metrics: Dictionary with metrics to save
            prefix: Optional prefix for file names
        """
        # Save in pickle format (for compatibility with existing code)
        with open(os.path.join(directory, f'{prefix}parameters.pickle'), 'wb') as f:
            pickle.dump(params, f)
        with open(os.path.join(directory, f'{prefix}metrics.pickle'), 'wb') as f:
            pickle.dump(metrics, f)
        
        # Filter metrics, excluding numpy arrays for the CSV
        metrics_readable = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        
        # Create a single CSV file with all information
        combined_data = {}
        # Add "param_" prefix to parameters to distinguish them
        combined_data.update({f"param_{k}": v for k, v in params.items()})
        # Add "metric_" prefix to metrics
        combined_data.update({f"metric_{k}": v for k, v in metrics_readable.items()})
        
        # Add date and time as reference
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        combined_data["timestamp"] = timestamp
        
        # Save to CSV
        combined_df = pd.DataFrame([combined_data])
        combined_df.to_csv(os.path.join(directory, f'{prefix}results.csv'), index=False)
        
        self.log('debug', f"Saved results in {directory}")
        

    def calculate_model_shap_umap(self, model, fold_idx, pool_method, target_label, output_dir, extract_umap):
        """
        Calculate and save SHAP values for a given classifier model using training data of a specific fold,
        with dynamic support for binary and multi-class classification.
        
        Args:
            model: Trained classifier instance with calculate_SHAP_values method.
            fold_idx: Fold index to retrieve training data.
            pool_method: Pooling method used.
            target_label: Target label for classification.
            output_dir: Directory to save SHAP outputs.
            extract_umap: Whether to extract UMAP visualization.
        
        Returns:
            dict: {
                'path': path to SHAP output directory,
                'shap_df': pd.DataFrame with SHAP importances.
            }
        """
        self.log('info', f"Calculating SHAP values for {target_label} using {pool_method} pooling (fold {fold_idx})")
        max_samples_shap = 50
        max_sample_umap = 2000
        
        # Retrieve training fold data
        X_train, X_val, y_train, y_val = self.sampler.get_fold_data(pool_method, target_label, fold_idx)
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])

        # Determine the number of classes
        unique_classes = sorted(y_combined.unique())
        n_classes = len(unique_classes)
        is_binary = n_classes == 2
        
        self.log('info', f"Detected {'binary' if is_binary else 'multi-class'} classification with {n_classes} classes: {unique_classes}")

        # Get balanced samples for SHAP calculation
        X_balanced_shap, y_balanced_shap = get_balanced_samples(X_combined, y_combined, max_samples_shap)

        # Calculate SHAP using classifier's method (saves plots and CSV)
        df_shap = model.calculate_SHAP_values(X_balanced_shap, output_dir)
        
        if extract_umap:
            # Feature selection based on SHAP values
            if is_binary:
                # For binary classification, use positive class SHAP values
                positive_class = max(unique_classes)  # Assume higher value is positive class
                shap_column = f'shap_class_{positive_class}'
                
                if shap_column in df_shap.columns:
                    df = df_shap.sort_values(shap_column, ascending=False).copy()
                    self.log('info', f"Using SHAP values for positive class ({positive_class}) for feature selection")
                else:
                    # Fallback: use first available SHAP column
                    shap_columns = [col for col in df_shap.columns if col.startswith('shap_class_')]
                    if shap_columns:
                        df = df_shap.sort_values(shap_columns[0], ascending=False).copy()
                        self.log('warning', f"Column {shap_column} not found, using {shap_columns[0]} instead")
                    else:
                        self.log('error', "No SHAP columns found in dataframe")
                        return {'path': output_dir, 'shap_df': df_shap}
            else:
                # For multi-class, use mean absolute SHAP values across all classes
                shap_columns = [col for col in df_shap.columns if col.startswith('shap_class_')]
                
                if shap_columns:
                    # Calculate mean absolute SHAP importance across all classes
                    df_shap['mean_abs_shap'] = df_shap[shap_columns].abs().mean(axis=1)
                    df = df_shap.sort_values('mean_abs_shap', ascending=False).copy()
                    self.log('info', f"Using mean absolute SHAP values across {len(shap_columns)} classes for feature selection")
                else:
                    self.log('error', "No SHAP columns found in dataframe")
                    return {'path': output_dir, 'shap_df': df_shap}
            
            # Select top features
            top_feat = df.head(50)['feature'].tolist()
            self.log('info', f"Top features selected with SHAP: {len(top_feat)}")
            
            if not top_feat:
                top_feat = df.iloc[:1]['feature'].tolist()
                self.log('warning', "No features found, using first feature as fallback")

            # Get balanced samples for UMAP
            X_balanced_umap, y_balanced_umap = get_balanced_samples(X_combined, y_combined, max_sample_umap)
            
            X_umap = X_balanced_umap[top_feat]
            
            # Scale features for UMAP
            X_scaled = StandardScaler().fit_transform(X_umap)
            X_for_umap = X_scaled

            # UMAP hyperparameter optimization
            best_score, best_params = -1, None
            self.log('info', "Optimizing UMAP hyperparameters...")
            
            for n in [15, 30, 50, 80, 100]:
                for d in [0, 0.1, 0.3, 0.5]:
                    reducer = umap.UMAP(
                        n_neighbors=n,
                        min_dist=d,
                        n_components=2,
                        metric='euclidean',
                        random_state=42
                    )
                    emb = reducer.fit_transform(X_for_umap)
                    
                    # Save intermediate UMAP projection
                    fig = plot_umap_projection(emb, y_balanced_umap, os.path.join(output_dir, f"umap_projection_{n}_{d}.png"))
                    
                    # Evaluate using silhouette score
                    score = silhouette_score(emb, y_balanced_umap)
                    if score > best_score:
                        best_score, best_params = score, (n, d)
            
            self.log('info', f"Best UMAP parameters: n_neighbors={best_params[0]}, min_dist={best_params[1]} (silhouette score: {best_score:.4f})")
            
            # Generate final UMAP with best parameters
            reducer = umap.UMAP(
                n_neighbors=best_params[0], 
                min_dist=best_params[1], 
                n_components=2, 
                metric='euclidean', 
                random_state=42
            )
            embedding = reducer.fit_transform(X_for_umap)
            
            # Save final UMAP projection
            fig = plot_umap_projection(
                embedding, 
                y_balanced_umap, 
                os.path.join(output_dir, f"umap_projection_best_{best_params[0]}_{best_params[1]}.png")
            )
            
            self.log('info', f"UMAP visualization saved with {n_classes} classes")

        return {'path': output_dir, 'shap_df': df_shap}