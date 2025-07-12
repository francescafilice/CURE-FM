from typing import Optional

from .BasePreprocessor import BasePreprocessor
from .Code15ECGFMPreprocessor import Code15ECGFMPreprocessor
from .Code15HuBERTECGPreprocessor import Code15HuBERTECGPreprocessor


class PreprocessorFactory:
    """
    Factory for creating preprocessors based on model and dataset type.
    Simply selects the appropriate preprocessor based on configuration.
    """
    
    @staticmethod
    def create_preprocessor(
        model_type: str,
        dataset_type: str,
        raw_data_path: str,
        processed_data_dir: str,
        ecg_dataset_path: str,
        meta_dataset_path: str,
        logger: Optional[object] = None,
        **kwargs
    ) -> BasePreprocessor:
        """
        Create an appropriate preprocessor based on the provided parameters.
        
        Args:
            model_type: Type of model (e.g., "ECG-FM")
            dataset_type: Type of dataset (e.g., "code15")
            raw_data_path: Path to raw data
            processed_data_dir: Directory for processed data
            ecg_dataset_path: Path to save the ECG dataset (optional)
            meta_dataset_path: Path to save metadata (optional)
            **kwargs: Additional parameters
            
        Returns:
            An appropriately configured BasePreprocessor instance
            
        Raises:
            ValueError: If the combination of model_type and dataset_type is not supported
        """
        
        # Normalize model and dataset types
        model_type = model_type.lower()
        dataset_type = dataset_type.lower()
        
        # Select the appropriate preprocessor based on model and dataset type
        if dataset_type == 'code15':
            if model_type in ['ecg-fm', 'ecgfm', 'ecg_fm']:
                return Code15ECGFMPreprocessor(
                    raw_data_path=raw_data_path,
                    processed_data_dir=processed_data_dir,
                    ecg_dataset_path=ecg_dataset_path,
                    meta_dataset_path=meta_dataset_path,
                    logger=logger,
                    **kwargs
                )
            elif any(model_type.startswith(prefix) for prefix in ['hubert-ecg', 'hubertecg', 'hubert_ecg']):
                # Extract metadata from kwargs for HuBERT-ECG preprocessor
                metadata = kwargs.pop('metadata', None)
                
                return Code15HuBERTECGPreprocessor(
                    raw_data_path=raw_data_path,
                    processed_data_dir=processed_data_dir,
                    ecg_dataset_path=ecg_dataset_path,
                    meta_dataset_path=meta_dataset_path,
                    logger=logger,
                    metadata=metadata, 
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported combination: model_type={model_type} for dataset_type={dataset_type}")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
