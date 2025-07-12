from typing import Optional

from .BaseFinetuner import BaseFinetuner
from .ECGFMFinetuner import ECGFMFinetuner
from .HuBERTFinetuner import HuBERTFinetuner


class FinetunerFactory:
    """
    Factory for creating finetuners based on model and dataset type.
    """
    
    @staticmethod
    def create_finetuner(
        model_type: str,
        dataset_type: str,
        processed_data_dir: str,
        checkpoint_path: str,
        save_dir: str,
        logger: Optional[object] = None,
        ecg_dataset_path: Optional[str] = None,
        meta_dataset_path: Optional[str] = None,
        **kwargs
    ) -> BaseFinetuner:
        """
        Create an appropriate finetuner based on the provided parameters.
        
        Args:
            model_type: Type of model (e.g., "ECG-FM", "HuBERT-ECG")
            dataset_type: Type of dataset (e.g., "code15")
            processed_data_dir: Directory containing processed data
            checkpoint_path: Path to the model checkpoint
            save_dir: Directory to save finetuned model checkpoints
            logger: Logger instance for tracking progress
            ecg_dataset_path: Path to the ECG dataset
            meta_dataset_path: Path to the metadata dataset
            **kwargs: Additional parameters
            
        Returns:
            An appropriately configured BaseFinetuner instance
            
        Raises:
            ValueError: If the combination of model_type and dataset_type is not supported
        """
        # Normalize model and dataset types
        model_type = model_type.lower()
        dataset_type = dataset_type.lower()
        
        # Set default paths if not provided
        if ecg_dataset_path is None:
            ecg_dataset_path = f"{processed_data_dir}/ecg_org_dataset.pkl"
        if meta_dataset_path is None:
            meta_dataset_path = f"{processed_data_dir}/meta_dataset.csv"
        
        # Select the appropriate finetuner based on model and dataset type
        if model_type in ['ecg-fm', 'ecgfm', 'ecg_fm']:
            return ECGFMFinetuner(
                processed_data_dir=processed_data_dir,
                ecg_dataset_path=ecg_dataset_path,
                meta_dataset_path=meta_dataset_path,
                checkpoint_path=checkpoint_path,
                save_dir=save_dir,
                dataset_name=dataset_type,
                logger=logger,
                **kwargs
            )
        elif model_type in ['hubert-ecg', 'hubert_ecg']:
            return HuBERTFinetuner(
                processed_data_dir=processed_data_dir,
                ecg_dataset_path=ecg_dataset_path,
                meta_dataset_path=meta_dataset_path,
                checkpoint_path=checkpoint_path,
                save_dir=save_dir,
                dataset_name=dataset_type,
                logger=logger,
                **kwargs
            )
        # Add more model types here as needed
        else:
            raise ValueError(f"Unsupported model type for finetuning: {model_type}")
