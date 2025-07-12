import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import copy
from .BaseClassifier import BaseClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
)
import shap


class SimpleDenseNN(nn.Module):
    """
    Implementation of a simple fully connected neural network.
    """
    
    def __init__(self, input_size, num_classes=2, hidden_layers=1, hidden_units=64, activation_fn=nn.ReLU):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of the input features
            num_classes: Number of output classes (2 for binary, >2 for multiclass)
            hidden_layers: Number of hidden layers
            hidden_units: Units for each hidden layer
            activation_fn: Activation function
        """
        super(SimpleDenseNN, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        
        # Building layers
        # First layer: input -> hidden
        self.layers.append(nn.Linear(input_size, hidden_units))
        self.layers.append(activation_fn())
        
        # Intermediate hidden layers if present
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.layers.append(activation_fn())
        
        # Output layer 
        # for binary classification, we set 2 output nodes (logits for both classes) - instead of the traditional only node -
        # so that we formulate the problem in terms of the softmax function, included in the CrossEntropyLoss function
        self.layers.append(nn.Linear(hidden_units, num_classes))
        

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Network output (logits)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def predict_proba(self, x):
        """
        Get probability outputs using softmax.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability outputs
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


class NNClassifier(BaseClassifier):
    """
    Implementation of a PyTorch neural network based classifier.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the neural network classifier.
        
        Args:
            logger: Logger instance for tracking progress
        """
        super().__init__("Neural Network", logger=logger)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = None
        
        self.log('info', f"Using device: {self.device}")
    

    def create_estimator(self, params=None):
        """
        Create an instance of the neural network model.
        
        Args:
            params: Parameters for the neural network (optional)
            
        Returns:
            Instance of SimpleDenseNN
        """
        # Parameters are handled differently for neural networks
        # So this function is mainly used for compatibility with BaseClassifier
        return None
    

    def _to_tensor(self, data):
        """
        Convert data to PyTorch tensors.
        
        Args:
            data: Data to convert
            
        Returns:
            PyTorch tensor
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32).to(self.device)
    

    def train(self, X, y, param_grid=None):
        """
        Train the model with hyperparameter search.
        
        Args:
            X: Training data
            y: Training labels
            param_grid: Grid of parameters to test, can also contain training parameters such as epochs and val_split
            
        Returns:
            Training results
        """
        self.log('info', f"Training neural network with {len(X)} samples")
        
        # Determine number of classes
        self.num_classes = len(np.unique(y))
        self.log('info', f"Detected {self.num_classes} classes")
        
        if self.logger:
            self.logger.start_timer("nn_train")
        
        # Fit the scaler on training data
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        
        # Default parameter grid if not specified
        if param_grid is None:
            param_grid = {
                'hidden_layers': [1, 2],
                'hidden_units': [64, 128],
                'learning_rates': [0.001, 0.01],
                'activation': ['relu', 'tanh'],
                'epochs': 50,
                'val_split': 0.2
            }
            self.log('debug', f"Using default parameter grid: {param_grid}")
        else:
            self.log('debug', f"Using provided parameter grid: {param_grid}")
        
        # Extract training parameters from param_grid
        epochs = param_grid.pop('epochs', 50) if 'epochs' in param_grid else 50
        val_split = param_grid.pop('val_split', 0.2) if 'val_split' in param_grid else 0.2
        batch_size = param_grid.pop('batch_size', 64) if 'batch_size' in param_grid else 64
        
        self.log('debug', f"Training parameters: epochs={epochs}, val_split={val_split}, batch_size={batch_size}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_split, random_state=42, stratify=y
        )
        
        self.log('debug', f"Split data: {len(X_train)} training samples, {len(X_val)} validation samples")
        
        # Prepare parameter combinations
        param_combinations = []
        for hl in param_grid.get('hidden_layers', [1]):
            for hu in param_grid.get('hidden_units', [64]):
                for lr in param_grid.get('learning_rates', [0.001]):
                    for act in param_grid.get('activation', ['relu']):
                        param_combinations.append({
                            'hidden_layers': hl,
                            'hidden_units': hu,
                            'learning_rates': lr,
                            'activation': act,
                            'input_size': X_train.shape[1],
                            'num_classes': self.num_classes
                        })
        
        self.log('info', f"Testing {len(param_combinations)} parameter combinations")
        
        # Search for the best combination of parameters
        best_score = 0
        best_model = None
        best_params = None
        best_metrics = None
        
        for idx, params in enumerate(param_combinations):
            self.log('info', f"Configuration {idx+1}/{len(param_combinations)}: {params}")
            
            # Create the model
            activation_fn = nn.ReLU if params['activation'] == 'relu' else nn.Tanh
            model = SimpleDenseNN(
                input_size=X_train.shape[1],
                num_classes=self.num_classes,
                hidden_layers=params['hidden_layers'],
                hidden_units=params['hidden_units'],
                activation_fn=activation_fn
            ).to(self.device)
            
            # Configure optimizer and loss 
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rates'])
            loss_fn = nn.CrossEntropyLoss()
            
            # Train the model
            model, val_metrics, history = self._train_model(
                model, optimizer, loss_fn, 
                X_train, y_train, 
                X_val, y_val, 
                epochs, batch_size
            )
            
            # Update the best model if necessary
            if val_metrics['f1_score'] > best_score:
                best_score = val_metrics['f1_score']
                best_model = copy.deepcopy(model)
                best_params = params.copy()
                best_params['epochs'] = epochs
                best_params['val_split'] = val_split
                best_params['batch_size'] = batch_size
                best_metrics = val_metrics
                
                self.log('info', f"New best f1_score: {val_metrics['f1_score']:.4f}")
            else:
                self.log('info', f"f1_score with this configuration: {val_metrics['f1_score']:.4f}")
        
        # Save the best model
        self.model = best_model
        self.best_model = best_model
        self.best_params = best_params
        self.is_fitted = True
        
        self.log('info', f"Best parameters: {best_params}")
        self.log('info', f"Best accuracy: {best_score:.4f}")
        
        # Log training results
        if self.logger:
            self.logger.stop_timer("nn_train")
            metrics_to_log = {
                "Best Parameters": str(best_params),
                "Best Accuracy": f"{best_score:.4f}",
                "Best F1 Score": f"{best_metrics['f1_score']:.4f}",
                "Training Device": str(self.device),
                "Network Architecture": f"{best_params['hidden_layers']} layers, {best_params['hidden_units']} units",
                "Training Samples": len(X_train),
                "Validation Samples": len(X_val),
                "Number of Classes": self.num_classes
            }
            self.logger.log_result("Neural Network Training Results", metrics_to_log)
            
            # Log memory usage
            self.logger.log_memory_usage()
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score
        }
    
    
    def _train_model(self, model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs, batch_size=64):
        """
        Train the neural network model with batch processing and early stopping.
        
        Args:
            model: The model to train
            optimizer: The optimizer
            loss_fn: The loss function
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs
            batch_size: Batch size for training
            
        Returns:
            Trained model, validation metrics, and training history
        """
        self.log('debug', f"Starting model training for {epochs} epochs with batch size {batch_size}")
        
        train_losses = []
        val_losses = []
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(self.device)
        
        # Create data loaders for batch processing
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model weights
        model.initialize_weights()
        
        # Variables for early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = epoch_train_loss / batch_count
            train_losses.append(avg_train_loss)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = loss_fn(val_outputs, y_val_tensor).item()
                val_losses.append(val_loss)
                
                # Check if this is the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                self.log('debug', f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                self.log('info', f"Early stopping at epoch {epoch+1}")
                break
        
        # Load the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.log('debug', "Loaded the best model state based on validation loss")
        
        # Calculate validation metrics for the best model
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_probs = torch.softmax(val_outputs, dim=1)
            val_preds = val_probs.cpu().numpy()
            
        if hasattr(y_val, 'values'):
            y_val = y_val.values
        else:
            y_val = y_val
            
        val_metrics = self._calculate_metrics(y_val, val_preds)
        
        # Log final metrics
        self.log('info', f"Training completed. Metrics:")
        for metric, value in val_metrics.items():
            self.log('info', f"  - {metric}: {value:.4f}")
        
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        
        return model, val_metrics, history
    

    def _predict_proba_batched(self, X, batch_size=1000):
        """
        Predict probabilities in batches to avoid memory issues.
        
        Args:
            X: Preprocessed data
            batch_size: Size of each batch
            
        Returns:
            Predicted probabilities for all classes
        """
        self.log('debug', f"Batched prediction with batch size {batch_size}")
        
        all_preds = []
        n_samples = X.shape[0]
        
        # Convert to tensor dataset for batch processing
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Track progress for large datasets
        num_batches = len(dataloader)
        self.log('debug', f"Predicting in {num_batches} batches")
        
        self.best_model.eval()
        with torch.no_grad():
            for i, (batch_X,) in enumerate(dataloader):
                if i % 10 == 0 or i == num_batches - 1:
                    self.log('debug', f"Processing batch {i+1}/{num_batches}")
                    
                batch_X = batch_X.to(self.device)
                batch_logits = self.best_model(batch_X)
                batch_probs = torch.softmax(batch_logits, dim=1)
                all_preds.append(batch_probs.cpu().numpy())
        
        return np.vstack(all_preds)
    
    
    def _predict_proba(self, X):
        """
        Predict probabilities for the given data.
        
        Args:
            X: Preprocessed data
            
        Returns:
            Predicted probabilities for all classes
        """
        self.log('debug', f"Predicting for {len(X)} samples")
        
        # Process in batches if the dataset is large
        if X.shape[0] > 1000:
            self.log('debug', f"Large dataset detected, using batched prediction")
            return self._predict_proba_batched(X, batch_size=1000)
        
        X_tensor = self._to_tensor(X)
        
        self.best_model.eval()  
        with torch.no_grad():
            logits = self.best_model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    
    def _calculate_metrics(self, y_true, y_pred_proba, threshold=0.5):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes])
            threshold: Threshold for binary classification (ignored for multiclass)
            
        Returns:
            Dictionary with metrics
        """
        # Convert probabilities to classes
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        
        if self.num_classes == 2:
            # Binary classification - use positive class probabilities
            y_pred_proba_pos = y_pred_proba[:, 1]
            y_pred = (y_pred_proba_pos >= threshold).astype(int)
        else:
            # Multiclass classification - use argmax
            y_pred_proba_pos = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else None
            y_pred = np.argmax(y_pred_proba, axis=1)
    
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle different averaging strategies based on number of classes
        # Binary classification - use traditional metrics
        if self.num_classes == 2:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        # multiclass classification - use weighted metrics
        else:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUROC 
        try:
            if self.num_classes == 2:
                auroc = roc_auc_score(y_true, y_pred_proba_pos)
            else:
                auroc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auroc = 0
            self.log('warning', "Could not calculate AUROC")
            
        # AUPRC 
        try:
            if self.num_classes == 2:
                auprc = average_precision_score(y_true, y_pred_proba_pos)
            else:
                auprc = average_precision_score(y_true, y_pred_proba, average='weighted')
        except:
            auprc = 0
            self.log('warning', "Could not calculate AUPRC")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auroc': auroc,
            'auprc': auprc
        }
    
    
    def load_model(self, input_path):
        """
        Load a model from disk.
        
        Args:
            input_path: Path of the saved model
        """
        self.log('info', f"Loading neural network model from {input_path}")
        
        try:
            model_data = torch.load(input_path, map_location=self.device)
            
            # Recreate the model
            activation_fn = nn.ReLU if model_data['params']['activation'] == 'relu' else nn.Tanh
            
            self.best_model = SimpleDenseNN(
                input_size=model_data['params'].get('input_size'),
                num_classes=model_data['params'].get('num_classes', 2),
                hidden_layers=model_data['params']['hidden_layers'],
                hidden_units=model_data['params']['hidden_units'],
                activation_fn=activation_fn
            ).to(self.device)
            
            # Load model weights
            self.best_model.load_state_dict(model_data['model_state_dict'])
            self.best_params = model_data['params']
            self.num_classes = model_data['params'].get('num_classes', 2)
            self.scaler = model_data['scaler']
            self.model = self.best_model
            self.is_fitted = True
            
            self.log('info', "Model loaded successfully")
            self.log('debug', f"Model parameters: {self.best_params}")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)


    def save_model(self, output_path):
        """
        Save the model to disk.
        
        Args:
            output_path: Path where to save the model
        """
        if not self.is_fitted:
            error_msg = "Model must be trained before saving"
            self.log('error', error_msg)
            raise ValueError(error_msg)
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.log('info', f"Saving neural network model to {output_path}")
        
        # Save the PyTorch model along with other data
        model_data = {
            'model_state_dict': self.best_model.state_dict(),
            'model_class': type(self.best_model).__name__,
            'params': self.best_params,
            'scaler': self.scaler
        }
        
        torch.save(model_data, output_path)
        self.log('info', "Model saved successfully")


    def _get_shap_values(self, X_scaled):
        """
        Calculate SHAP values for the Neural Network model.
        
        Args:
            X_scaled: Standardized data
            
        Returns:
            List of SHAP values for each class
        """
        device = next(self.model.parameters()).device
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        # Choose a background subset for DeepExplainer
        n_bg = min(100, X_tensor.shape[0])
        idx_bg = np.random.choice(X_tensor.shape[0], n_bg, replace=False)
        background = X_tensor[idx_bg]

        # Create DeepExplainer
        explainer = shap.DeepExplainer(self.model, background)

        # Calculate SHAP values
        shap_vals = explainer.shap_values(X_tensor)
        
        # Debug: Log the structure of SHAP values
        if isinstance(shap_vals, list):
            self.log('debug', f"SHAP values returned as list with {len(shap_vals)} elements")
            for i, sv in enumerate(shap_vals):
                self.log('debug', f"  Element {i} shape: {sv.shape}")
        else:
            self.log('debug', f"SHAP values returned as single array with shape: {shap_vals.shape}")
        
        # Handle different SHAP output formats to always return a list
        if isinstance(shap_vals, list):
            # Already a list of arrays for each class
            return shap_vals
        elif isinstance(shap_vals, np.ndarray):
            if len(shap_vals.shape) >= 3:
                # Multi-class output: shape (n_samples, n_features, n_classes)
                return [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
            elif len(shap_vals.shape) == 2 and shap_vals.shape[1] == X_tensor.shape[1]:
                # Binary classification: shape (n_samples, n_features)
                # For binary classification, SHAP returns values for positive class
                # Create values for negative class (inverse)
                return [-shap_vals, shap_vals]
            elif len(shap_vals.shape) == 2 and shap_vals.shape[1] == 2:
                # This means SHAP returned (n_samples, 2) - this is wrong for our purposes
                # This should not happen with DeepExplainer for 2-class output
                error_msg = f"Unexpected SHAP format: got shape {shap_vals.shape}, expected (n_samples, n_features) or list of such arrays. This suggests SHAP is explaining output logits instead of input features."
                self.log('error', error_msg)
                raise ValueError(error_msg)
            else:
                # Other unexpected formats
                self.log('warning', f"Unexpected SHAP values shape: {shap_vals.shape}")
                return [shap_vals]
        else:
            self.log('warning', "Unexpected SHAP values format")
            return [shap_vals]