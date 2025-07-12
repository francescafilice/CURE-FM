import numpy as np
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from .BaseClassifier import BaseClassifier
import gc
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
import shap

class XGBoostClassifier(BaseClassifier):
    """
    Implementation of XGBoost classifier with automatic GPU support 
    and memory-efficient training optimization.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the XGBoost classifier with auto GPU detection.
        
        Args:
            logger: Logger instance for tracking progress
        """
        super().__init__("XGBoost", logger=logger)
        self.batch_size = 10000  # Standard batch size
        
        # Auto-detect GPU availability
        self.use_gpu = self._check_gpu_availability()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.log('info', f"Using {self.device.upper()} for XGBoost")
    

    def _check_gpu_availability(self):
        """Check if GPU acceleration is available for XGBoost."""
        try:
            # Test GPU availability with minimal data
            test_data = np.random.rand(10, 10)
            test_label = np.random.randint(0, 2, 10)
            test_dmatrix = xgb.DMatrix(test_data, label=test_label)
            
            # Try with GPU
            params = {'tree_method': 'hist', 'device': 'cuda'}
            model = xgb.train(params, test_dmatrix, num_boost_round=1)
            self.log('info', "XGBoost GPU acceleration is available")
            return True
        except Exception as e:
            self.log('warning', f"GPU acceleration not available: {e}")
            self.log('info', "Falling back to CPU")
            return False
    

    def create_estimator(self, params=None, num_classes=None):
        """
        Create an instance of the XGBoost model.
        
        Args:
            params: Parameters for the XGBoost model (optional)
            num_classes: Number of classes (optional, auto-detected if not provided)
            
        Returns:
            Instance of XGBClassifier
        """
        default_params = {
            "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
            "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
            "tree_method": "hist",
            "device": self.device,
            "random_state": 42
        }
    
        if num_classes > 2:
            default_params["num_class"] = num_classes
    
        all_params = {**default_params, **(params or {})}
        self.log('debug', f"Creating XGBoost model with parameters: {all_params}")
        return xgb.XGBClassifier(**all_params)
    

    def train(self, X, y, param_grid=None):
        """
        Train the model using memory-efficient approach with parameter search.
        
        Args:
            X: Training data
            y: Training labels
            param_grid: Grid of parameters to test (optional)
            
        Returns:
            Dictionary with training results
        """
        self.log('info', f"Training XGBoost model with {len(X)} samples")
        
        # Detect number of classes from target labels
        unique_classes = np.unique(y)
        self.num_classes = len(unique_classes)
        self.log('info', f"Detected {self.num_classes} classes: {unique_classes}")
        
        if self.logger:
            self.logger.start_timer("xgboost_train")
        
        # Fit the scaler on training data
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        
        # Default parameter grid if not specified
        if param_grid is None:
            param_grid = {
                'n_estimators': [100],
                'max_depth': [6],
                'learning_rate': [0.3],
                'subsample': [1.0],
                'colsample_bytree': [1.0],
                'min_child_weight': [1]
            }
            self.log('debug', f"Using default parameter grid: {param_grid}")
        else:
            self.log('debug', f"Using provided parameter grid: {param_grid}")
        
        # Always use memory efficient training
        self.log('info', f"Using memory-efficient training with batch size {self.batch_size}")
        best_model, best_params, best_score = self._memory_efficient_training(
            X_scaled, y, param_grid)
        
        gc.collect()  # Free memory
        
        # Stop timer
        if self.logger:
            self.logger.stop_timer("xgboost_train")
            self.logger.log_result("XGBoost Training Results", {
                "Best Parameters": str(best_params),
                "Best Validation Score": f"{best_score:.4f}",
                "Training Device": self.device.upper(),
                "Training Samples": len(X),
                "Feature Count": X.shape[1],
                "Number of Classes": self.num_classes
            })
        
        self.log('info', f"XGBoost training completed. Best score: {best_score:.4f}")
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score
        }
    

    def _memory_efficient_training(self, X, y, param_grid):
        """
        Train the model using a memory-efficient approach.
        
        Args:
            X: Training data
            y: Training labels
            param_grid: Grid of parameters to test
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        self.log('info', f"Using memory-efficient training with batch size {self.batch_size}")
        
        # Create a validation set (20%)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.log('debug', f"Split data: {len(X_train)} training samples, {len(X_val)} validation samples")
        
        # Convert validation data to DMatrix for efficient evaluation
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Parameter grid search
        param_combinations = list(ParameterGrid(param_grid))
        self.log('info', f"Testing {len(param_combinations)} parameter combinations")

        # Variables to track the best model
        best_score = float('-inf')
        best_model = None
        best_params = None
        
        # Select appropriate eval_metric based on number of classes
        is_multiclass = self.num_classes > 2
        eval_metric = "mlogloss" if is_multiclass else "auc"
        objective = "multi:softprob" if is_multiclass else "binary:logistic"
        
        for i, orig_params in enumerate(param_combinations):
            self.log('info', f"Trying parameters {i+1}/{len(param_combinations)}: {orig_params}")
            
            # Create a copy and extract num_boost_round
            params = orig_params.copy()
            num_boost_round = params.pop('n_estimators', 100)
            
            # Set up XGBoost parameters
            params.update({
                "tree_method": "hist",
                "device": self.device,
                "objective": objective,
                "eval_metric": eval_metric
            })
            if is_multiclass:
                params["num_class"] = self.num_classes
            
            # Process data in batches
            num_batches = (len(X_train) + self.batch_size - 1) // self.batch_size
            model = None
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(X_train))
                
                batch_msg = f"Processing batch {batch_idx+1}/{num_batches}"
                if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                    self.log('debug', batch_msg)
                
                # Extract batch data
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
                
                # Train or update model
                if batch_idx == 0:
                    model = xgb.train(
                        params,
                        dtrain_batch,
                        num_boost_round=num_boost_round,
                        evals=[(dval, 'validation')],
                        early_stopping_rounds=20,
                        verbose_eval=False
                    )
                else:
                    model = xgb.train(
                        params,
                        dtrain_batch,
                        num_boost_round=num_boost_round,
                        xgb_model=model,
                        evals=[(dval, 'validation')],
                        early_stopping_rounds=20,
                        verbose_eval=False
                    )
                
                # Clean memory
                del X_batch, y_batch, dtrain_batch
                gc.collect()
            
            # Evaluate the model with appropriate metric
            val_preds = model.predict(dval)
            if is_multiclass:
                # Use mlogloss for multiclass
                val_score = self._calculate_multiclass_score(y_val, val_preds)
                self.log('info', f"Validation Score (multiple log loss): {val_score:.4f}")
            else:
                # Use AUC for binary
                val_score = self._calculate_auc(y_val, val_preds)
                self.log('info', f"Validation AUC: {val_score:.4f}")
            
            # Update if this is the best model
            if val_score > best_score:
                best_score = val_score
                best_params = orig_params.copy()
                
                # Convert Booster to XGBClassifier
                best_model = xgb.XGBClassifier(
                    tree_method="hist", 
                    device=self.device
                )
                best_model._Booster = model
                best_model.n_classes_ = self.num_classes
                
                self.log('info', f"New best score: {best_score:.4f}")
            
            # Clean up
            del model, val_preds
            gc.collect()
        
        # Save best model
        self.best_model = best_model
        self.best_params = best_params
        self.model = best_model
        self.is_fitted = True
        
        return best_model, best_params, best_score
    

    def _calculate_auc(self, y_true, y_pred):
        """Calculate AUC score."""
        return roc_auc_score(y_true, y_pred)
    

    def _calculate_multiclass_score(self, y_true, y_pred):
        """Calculate negative mlogloss score for multiclass problems."""
        loss = log_loss(y_true, y_pred)
        return -loss  # Negate for maximization approach
    

    def _predict_proba(self, X):
        """
        Predict probabilities using batched approach.
        
        Args:
            X: Preprocessed data
            
        Returns:
            Predicted probabilities for the positive class
        """
        self.log('debug', f"Predicting for {len(X)} samples")
        # Always use batched prediction
        return self._predict_proba_batched(X)
    

    def _predict_proba_batched(self, X):
        """
        Predict probabilities in batches to avoid memory issues.
        
        Args:
            X: Preprocessed data
            
        Returns:
            Predicted probabilities for all classes
        """
        if not self.is_fitted:
            error_msg = "Model must be trained before prediction"
            self.log('error', error_msg)
            raise ValueError(error_msg)
            
        num_samples = X.shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        # For multiclass, we need probabilities for all classes
        is_multiclass = self.num_classes > 2
        all_probas = np.zeros((num_samples, self.num_classes)) if is_multiclass else np.zeros((num_samples, 1))
    
        self.log('debug', f"Predicting in {num_batches} batches")
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            X_batch = X[start_idx:end_idx]
            batch_probas = self.best_model.predict_proba(X_batch)
            all_probas[start_idx:end_idx] = batch_probas if is_multiclass else batch_probas[:, 1].reshape(-1, 1)
            
            del X_batch, batch_probas
            gc.collect()
    
        return all_probas
    

    def _get_shap_values(self, X_scaled):
        """
        Calculate SHAP values for the XGBoost model.
        
        Args:
            X_scaled: Standardized data
            
        Returns:
            List of SHAP values for each class
        """
        # Compute SHAP values
        explainer = shap.TreeExplainer(self.best_model)
        shap_vals = explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats to always return a list
        if isinstance(shap_vals, list):
            # Already a list of arrays for each class
            return shap_vals
        elif isinstance(shap_vals, np.ndarray):
            if len(shap_vals.shape) >= 3:
                # Multi-class output: shape (n_samples, n_features, n_classes)
                return [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
            elif len(shap_vals.shape) == 2:
                # Binary classification: shape (n_samples, n_features)
                # For binary classification, SHAP returns values for positive class
                # Create values for negative class (inverse)
                return [-shap_vals, shap_vals]
            else:
                self.log('warning', f"Unexpected SHAP values shape: {shap_vals.shape}")
                return [shap_vals]
        else:
            self.log('warning', "Unexpected SHAP values format")
            return [shap_vals]
