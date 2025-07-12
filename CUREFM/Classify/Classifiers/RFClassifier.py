from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
import shap
from .BaseClassifier import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """
    Implementation of a Random Forest classifier with automatic hyperparameter selection.
    Supports both binary and multiclass classification.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the Random Forest classifier.
        
        Args:
            logger: Logger instance for tracking progress
        """
        super().__init__("Random Forest", logger=logger)
        self.n_classes = None   # 2 if binary classification, >2 for multiclass
    
    def create_estimator(self, params=None):
        """
        Create an instance of the Random Forest model.
        
        Args:
            params: Parameters for Random Forest (optional)
            
        Returns:
            Instance of RandomForestClassifier
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 30,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        self.log('debug', f"Creating Random Forest with parameters: {default_params}")
        return RF(**default_params)
    
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
        self.log('info', f"Training Random Forest with {len(X)} samples")
        
        # Determine number of classes
        self.n_classes = len(np.unique(y))
        self.log('info', f"Detected {self.n_classes} classes for classification")
        
        # Timer start
        if self.logger:
            self.logger.start_timer("rf_train")
        
        # Fit the scaler on training data
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        
        # Default parameter grid if not specified
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            self.log('debug', f"Using default parameter grid: {param_grid}")
        else:
            self.log('debug', f"Using provided parameter grid: {param_grid}")
        
        # Execute grid search
        best_model, best_params, best_score = self.grid_search(X_scaled, y, param_grid)
     
        # Timer stop
        if self.logger:
            self.logger.stop_timer("rf_train")
            self.logger.log_result("Random Forest Training Results", {
                "Best Parameters": str(best_params),
                "Best CV Score": f"{best_score:.4f}",
                "Training Samples": len(X),
                "Feature Count": X.shape[1],
                "Number of Classes": self.n_classes
            })
        
        self.log('info', f"Random Forest training completed. Best score: {best_score:.4f}")
        
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
        Calculate SHAP values for the Random Forest model.
        
        Args:
            X_scaled: Standardized data
            
        Returns:
            List of SHAP values for each class
        """
        # Compute SHAP values
        explainer = shap.TreeExplainer(self.best_model)
        shap_vals = explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats
        if isinstance(shap_vals, list):
            # Already a list of arrays for each class
            return shap_vals
        elif isinstance(shap_vals, np.ndarray):
            # Convert to list of arrays for each class
            if len(shap_vals.shape) == 3:
                return [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
            else:
                # For binary output that's just a single array
                return [np.zeros_like(shap_vals), shap_vals]
        else:
            self.log('warning', "Unexpected SHAP values format")
            return [shap_vals]