from sklearn.svm import SVC
import numpy as np
from .BaseClassifier import BaseClassifier

class SVMClassifier(BaseClassifier):
    """
    Implementation of an SVM classifier with automatic hyperparameter selection.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the SVM classifier.
        
        Args:
            logger: Logger instance for tracking progress
        """
        super().__init__("SVM", logger=logger)
    
    def create_estimator(self, params=None):
        """
        Create an instance of the SVM model.
        
        Args:
            params: Parameters for SVM (optional)
            
        Returns:
            Instance of SVC
        """
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        self.log('debug', f"Creating SVM with parameters: {default_params}")
        return SVC(**default_params)
    
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
        self.log('info', f"Training SVM with {len(X)} samples")
        
        # Timer start
        if self.logger:
            self.logger.start_timer("svm_train")
        
        # Fit the scaler on training data
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        
        # Default parameter grid if not specified
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
            self.log('debug', f"Using default parameter grid: {param_grid}")
        else:
            self.log('debug', f"Using provided parameter grid: {param_grid}")
        
        # Execute grid search
        best_model, best_params, best_score = self.grid_search(X_scaled, y, param_grid)
        
        # Timer stop
        if self.logger:
            self.logger.stop_timer("svm_train")
            self.logger.log_result("SVM Training Results", {
                "Best Parameters": str(best_params),
                "Best CV Score": f"{best_score:.4f}",
                "Training Samples": len(X),
                "Feature Count": X.shape[1]
            })
        
        self.log('info', f"SVM training completed. Best score: {best_score:.4f}")
        
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
            Probabilities of the positive class
        """
        self.log('debug', f"Prediction on {len(X)} samples")
        probas = self.model.predict_proba(X)[:, 1]
        return probas.reshape(-1, 1) if len(probas.shape) == 1 else probas
    