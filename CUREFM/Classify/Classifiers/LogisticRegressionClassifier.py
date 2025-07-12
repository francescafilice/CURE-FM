from sklearn.linear_model import LogisticRegression 
import numpy as np
import shap
from .BaseClassifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """
    Implementation of a Logistic Regression classifier with automatic hyperparameter selection.
    Supports both binary and multiclass classification.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the Logistic Regression classifier.
        
        Args:
            logger: Logger instance for tracking progress
        """
        super().__init__("Logistic Regression", logger=logger)
        self.n_classes = None   # 2 if binary classification, >2 for multiclass
    
    def create_estimator(self, params=None):
        """
        Create an instance of the Logistic Regression model.
        
        Args:
            params: Parameters for Logistic Regression (optional)
            
        Returns:
            Instance of LogisticRegression
        """
        default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        self.log('debug', f"Creating Logistic Regression with parameters: {default_params}")
        return LogisticRegression(**default_params)
    
    def train(self, X, y, param_grid=None):
        """
        Train the model with automatic hyperparameter search.
        
        Args:
            X: Training data
            y: Training labels
            param_grid: Grid of parameters to test (optional)
            
        Returns:
            Training results
        """
        self.log('info', f"Training Logistic Regression with {len(X)} samples")
        
        # Determine number of classes
        self.n_classes = len(np.unique(y))
        self.log('info', f"Detected {self.n_classes} classes for classification")
        
        # Timer start
        if self.logger:
            self.logger.start_timer("lr_train")
        
        # Fit the scaler on training data
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        
        # Default parameter grid if not specified
        if param_grid is None:
            # Adjust solver and penalty based on problem type
            if self.n_classes > 2:
                # For multiclass, ensure compatible solver
                param_grid = {
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            else:
                # For binary, can use more solvers and penalties
                param_grid = {
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                }
            self.log('debug', f"Using default parameter grid: {param_grid}")
        else:
            self.log('debug', f"Using provided parameter grid: {param_grid}")
        
        # Execute grid search
        best_model, best_params, best_score = self.grid_search(X_scaled, y, param_grid)
     
        # Timer stop
        if self.logger:
            self.logger.stop_timer("lr_train")
            self.logger.log_result("Logistic Regression Training Results", {
                "Best Parameters": str(best_params),
                "Best CV Score": f"{best_score:.4f}",
                "Training Samples": len(X),
                "Feature Count": X.shape[1],
                "Number of Classes": self.n_classes
            })
        
        self.log('info', f"Logistic Regression training completed. Best score: {best_score:.4f}")
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score
        }
    
    def _predict_proba(self, X):
        """
        Predict probabilities for the given data.
        
        Args:
            X: Preprocessed data
            
        Returns:
            Probabilities for all classes
        """
        self.log('debug', f"Prediction on {len(X)} samples")
        probas = self.model.predict_proba(X)
        
        # For binary classification, return positive probability
        if self.n_classes == 2 and probas.shape[1] == 2:
            return probas[:, 1].reshape(-1, 1)
        
        # For multiclass, return all probabilities
        return probas
    
    def _get_shap_values(self, X_scaled):
        """
        Calculate SHAP values for the Logistic Regression model.
        
        Args:
            X_scaled: Standardized data
            
        Returns:
            List of SHAP values for each class
        """
        # Use Linear explainer for logistic regression
        explainer = shap.LinearExplainer(self.best_model, X_scaled)
        shap_vals = explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats
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