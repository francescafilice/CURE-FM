import os
import yaml
from typing import Dict, Any, Optional, List, Union


class ConfigReader:
    """
    Class for reading and managing YAML configuration files.
    Allows direct access to first-level attributes and provides dictionaries
    for nested parameters.
    """
    
    def __init__(self, config_path: str):
        """
        Initializes the configuration reader.
        
        Args:
            config_path: Path to the YAML configuration file
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the file is not a valid YAML
        """
        self.config_path = config_path
        self._attributes = {}  
        self._config = {}  
        
        self._load_config()
        self._create_directories()

    
    def _load_config(self) -> None:
        """
        Loads the configuration from the YAML file and sets first-level attributes.
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the YAML is not valid
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self._config = config
            
            for key, value in config.items():
                if not key.endswith('_params') and not isinstance(value, dict):
                    self._attributes[key] = value
            
            print(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {str(e)}")
        
    
    def _create_directories(self) -> None:
        """
        Creates the necessary directories specified in the configuration.
        """
        directories = [
            self.get('processed_data_dir'),
            self.get('embeddings_dir'),
            self.get('classification_output_dir'),
            os.path.dirname(self.get('ecg_dataset_path', '')),
            os.path.dirname(self.get('meta_dataset_path', ''))
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                print(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
    

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the configuration.
        
        Args:
            key: Key of the value to get
            default: Default value if the key doesn't exist
            
        Returns:
            The value associated with the key or the default
        """
        return self._config.get(key, default)
    

    def __getattr__(self, name: str) -> Any:
        """
        Handles access to attributes not found directly in the class.
        
        Args:
            name: Name of the attribute
            
        Returns:
            The value of the attribute or None if it doesn't exist
            
        Raises:
            AttributeError: If the attribute doesn't exist and is not a configuration parameter
        """
        # Check if it's a saved first-level attribute
        if name in self._attributes:
            return self._attributes[name]
            
        # Check if it's a parameter group (e.g. preprocessing_params)
        if name.endswith('_params') and name in self._config:
            return self._config[name]
        
        # Check if it's a first-level attribute in the configuration
        if name in self._config:
            return self._config[name]
        
        # Attribute not found
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    

    @property
    def preprocessing_params(self) -> Dict[str, Any]:
        """
        Gets preprocessing parameters as a dictionary.
        
        Returns:
            Dictionary with preprocessing parameters
        """
        return self._config.get('preprocessing_params', {})
    

    @property
    def embedding_params(self) -> Dict[str, Any]:
        """
        Gets parameters for embedding extraction as a dictionary.
        
        Returns:
            Dictionary with parameters for embedding extraction
        """
        return self._config.get('embedding_params', {})
    

    @property
    def classification_params(self) -> Dict[str, Any]:
        """
        Gets parameters for classification as a dictionary.
        
        Returns:
            Dictionary with parameters for classification
        """
        return self._config.get('classification_params', {})
    

    @property
    def data_sampler_params(self) -> Dict[str, Any]:
        """
        Gets parameters for data sampling as a dictionary.
        
        Returns:
            Dictionary with parameters for data sampling
        """
        return self._config.get('data_sampler_params', {})
    

    @property
    def pool_methods(self) -> List[str]:
        """
        Gets pooling methods as a list.
        
        Returns:
            List of pooling methods
        """
        return self._config.get('pool_methods', ['avg', 'max', 'min', 'lst'])
    

    def validate(self) -> bool:
        """
        Validates the configuration by checking essential parameters.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Check essential parameters
        essential_params = [
            'dataset_type',
            'model_type',
            'raw_data_path'
        ]
        
        for param in essential_params:
            if not self.get(param):
                print(f"Error: parameter '{param}' missing or empty in configuration")
                return False
        
        # Check that at least one phase is enabled
        if not (self.get('preprocess_dataset', True) 
                or self.get('extract_embeddings', True)
                or self.get('run_classification', False)):
            print("Warning: no phase enabled in configuration")
        
        # If embedding extraction is enabled, verify checkpoint path
        if self.get('extract_embeddings', True) and not self.get('ckpt_path'):
            print("Error: 'ckpt_path' missing, required for embedding extraction")
            return False
        
        # If classification is enabled, verify necessary parameters
        if self.get('run_classification', False):
            if not self.get('classification_params'):
                print("Error: 'classification_params' missing, required for classification")
                return False
            
            if not self.get('data_sampler_params'):
                print("Error: 'data_sampler_params' missing, required for classification")
                return False
        
        return True
    