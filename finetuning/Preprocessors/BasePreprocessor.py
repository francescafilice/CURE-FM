from abc import ABC, abstractmethod
import os
from typing import Optional


class BasePreprocessor(ABC):
    """
    Abstract base class for all ECG dataset preprocessors.
    Defines the common interface that all preprocessors must implement.
    """
    
    def __init__(self,
                 raw_data_path: str,
                 processed_data_dir: str,
                 ecg_dataset_path: str,
                 meta_dataset_path: str,
                 logger: Optional[object] = None,
                 **kwargs):
        """
        Initialize the preprocessor with the provided parameters.
        
        Args:
            raw_data_path: Path to raw data
            processed_data_dir: Directory to save processed data
            ecg_dataset_path: Path to save the ECG dataset
            meta_dataset_path: Path to save metadata
            **kwargs: Additional parameters
        """
        # Base paths
        self.package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Input data paths
        self.raw_data_path = raw_data_path
        
        # Processed output paths
        self.processed_data_dir = processed_data_dir
        self.ecg_dataset_path = ecg_dataset_path
        self.meta_dataset_path = meta_dataset_path
        
        # Simplified temp directory for all processors
        self.temp_dir = os.path.join(self.processed_data_dir, 'temp')
        
        # Logger
        self.logger = logger
        
        # Other parameters
        self.kwargs = kwargs
        
        # Create necessary directories
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.ecg_dataset_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_dataset_path), exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Processing script paths
        self.processor_dir = os.path.join(self.package_dir, 'processor')
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.debug(f"Raw data path: {self.raw_data_path}")
        self.logger.debug(f"Processed data directory: {self.processed_data_dir}")
        self.logger.debug(f"ECG dataset path: {self.ecg_dataset_path}")
        self.logger.debug(f"Metadata dataset path: {self.meta_dataset_path}")
    
    
    @abstractmethod
    def preprocess(self) -> None:
        """
        Execute complete dataset preprocessing.
        This method should be implemented by all subclasses.
        """
        pass
    
    @abstractmethod
    def prepare_raw_data(self) -> None:
        """
        Prepare raw data for preprocessing.
        """
        pass
    
    @abstractmethod
    def process_signals(self) -> None:
        """
        Process raw ECG signals.
        """
        pass
    
    @abstractmethod
    def process_metadata(self) -> None:
        """
        Process metadata associated with ECG signals.
        """
        pass
    
    @abstractmethod
    def compose_dataset(self) -> None:
        """
        Compose the final dataset by combining processed signals and metadata.
        Should handle all available labels, not just a single one.
        """
        pass
    
    @abstractmethod
    def clean_intermediate_files(self) -> None:
        """
        Remove intermediate files created during preprocessing.
        """
        pass
    
    def clean_temp_dir(self):
        """
        Clean the temporary directory.
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
            
        self.log('debug', "Temporary directory cleaned successfully")
    
    def save_preprocessing_info(self, output_path: Optional[str] = None) -> None:
        """
        Save preprocessing information to a YAML file.
        
        Args:
            output_path: Output file path
        """
        import yaml
        
        if output_path is None:
            output_path = os.path.join(self.processed_data_dir, 'preprocessing_info.yaml')
        
        preprocessing_info = {
            'raw_data_path': self.raw_data_path,
            'processed_data_dir': self.processed_data_dir,
            'ecg_dataset_path': self.ecg_dataset_path,
            'meta_dataset_path': self.meta_dataset_path,
            **self.kwargs
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(preprocessing_info, f)
        
        self.log('info', f"Preprocessing information saved to: {output_path}")
        
    def log(self, level, message):
        """
        Log a message if logger is available.
        
        Args:
            level: Log level (info, debug, warning, error, critical)
            message: Message to log
        """
        if not self.logger:
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