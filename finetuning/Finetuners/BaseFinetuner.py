from abc import ABC, abstractmethod
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import random


class BaseFinetuner(ABC):
    """
    Abstract base class for all ECG finetuning models.
    Defines the common interface that all finetuners must implement.
    """
    
    def __init__(self,
                 processed_data_dir: str,
                 ecg_dataset_path: str,
                 meta_dataset_path: str,
                 checkpoint_path: str,
                 save_dir: str,
                 logger=None,
                 **kwargs):
        """
        Initialize the finetuner with the provided parameters.
        
        Args:
            processed_data_dir: Directory containing processed data
            ecg_dataset_path: Path to the ECG dataset
            meta_dataset_path: Path to the metadata dataset
            checkpoint_path: Path to the pretrained model checkpoint
            save_dir: Directory to save finetuned model checkpoints
            logger: Logger instance for tracking progress
            **kwargs: Additional parameters
        """
        # Base paths
        self.package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Input data paths
        self.processed_data_dir = processed_data_dir
        self.ecg_dataset_path = ecg_dataset_path
        self.meta_dataset_path = meta_dataset_path
        
        # Model paths
        self.checkpoint_path = checkpoint_path
        self.save_dir = save_dir
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Logger
        self.logger = logger
        
        # Other parameters
        self.kwargs = kwargs
        
        # Device
        self.device_string = kwargs.get('finetuning_params', {}).get('device', 'cuda0')
        self.device = torch.device(self.device_string if torch.cuda.is_available() else 'cpu')
        
        # Internal state
        self.model = None
        self.data_sampler = None
        self.target_label = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Log initialization
        self.log('info', f"Initialized {self.__class__.__name__}")
        self.log('debug', f"Processed data directory: {self.processed_data_dir}")
        self.log('debug', f"ECG dataset path: {self.ecg_dataset_path}")
        self.log('debug', f"Metadata dataset path: {self.meta_dataset_path}")
        self.log('debug', f"Checkpoint path: {self.checkpoint_path}")
        self.log('debug', f"Save directory: {self.save_dir}")
        self.log('debug', f"Using device: {self.device}")
    
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
    
    @abstractmethod
    def build_model(self) -> None:
        """
        Build the foundation model for finetuning.
        """
        pass
    
    @abstractmethod
    def prepare_data(self, target_label: str, batch_size: int = 32) -> None:
        """
        Prepare the data for finetuning.
        
        Args:
            target_label: Target label for binary classification
            batch_size: Batch size for training
        """
        pass
    
    @abstractmethod
    def finetune(self, epochs: int = 10, learning_rate: float = 1e-4) -> None:
        """
        Finetune the model on the prepared data.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the finetuned model on the test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def process_all(self, target_label: str, batch_size: int, 
                    epochs: int, learning_rate: float) -> Dict[str, float]:
        """
        Process all finetuning steps: build model, prepare data, finetune, and evaluate.
        
        Args:
            target_label: Target label for binary classification
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.log('info', "Starting finetuning process...")
        
        # Build model
        self.log('info', "Building model...")
        if self.logger:
            self.logger.start_timer("build_model")
        
        self.build_model()
        
        if self.logger:
            self.logger.stop_timer("build_model")
        
        # Prepare data
        self.log('info', f"Preparing data for target label: {target_label}")
        if self.logger:
            self.logger.start_timer("prepare_data")
        
        self.prepare_data(target_label=target_label, batch_size=batch_size)
        
        if self.logger:
            self.logger.stop_timer("prepare_data")
        
        # Finetune model
        self.log('info', f"Finetuning model with {epochs} epochs and learning rate {learning_rate}")
        if self.logger:
            self.logger.start_timer("finetune")
        
        self.finetune(epochs=epochs, learning_rate=learning_rate)
        
        if self.logger:
            self.logger.stop_timer("finetune")
        
        # Evaluate model
        self.log('info', "Evaluating finetuned model...")
        if self.logger:
            self.logger.start_timer("evaluate")
        
        metrics = self.evaluate()
        
        if self.logger:
            self.logger.stop_timer("evaluate")
        
        self.log('info', f"Finetuning completed successfully. Metrics: {metrics}")
        
        # Save finetuning information
        self.save_finetuning_info()
        
        # Log memory usage
        if self.logger:
            self.logger.log_memory_usage()
            
        return metrics
    
    def save_finetuning_info(self, output_path: Optional[str] = None) -> None:
        """
        Save finetuning information to a YAML file.
        
        Args:
            output_path: Output file path
        """
        import yaml
        
        if output_path is None:
            output_path = os.path.join(self.save_dir, 'finetuning_info.yaml')
        
        self.log('info', f"Saving finetuning information to {output_path}")
        
        finetuning_info = {
            'processed_data_dir': self.processed_data_dir,
            'ecg_dataset_path': self.ecg_dataset_path,
            'meta_dataset_path': self.meta_dataset_path,
            'checkpoint_path': self.checkpoint_path,
            'save_dir': self.save_dir,
            'target_label': self.target_label,
            'device': str(self.device),
            **self.kwargs
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(finetuning_info, f)
        
        self.log('info', f"Finetuning information saved to: {output_path}")
        