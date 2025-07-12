from typing import Optional

from .BaseEmbedder import BaseEmbedder
from .ECGFMEmbedder import ECGFMEmbedder
from .HuBERTECGEmbedder import HuBERTECGEmbedder


class EmbedderFactory:
    """
    Factory for creating embedders based on model and dataset type.
    """
    
    @staticmethod
    def create_embedder(
        model_type: str,
        dataset_type: str,
        id_column: str,
        processed_data_dir: str,
        output_dir: str,
        checkpoint_path: str,
        ecg_dataset_path: str,
        meta_dataset_path: str,
        logger: Optional[object] = None,
        **kwargs
    ) -> BaseEmbedder:
        """
        Create an appropriate embedder based on the provided parameters.
        
        Args:
            model_type: Type of model (e.g., "ECG-FM", "HuBERT-ECG")
            dataset_type: Type of dataset (e.g., "code15")
            id_column: Column name for the ID in the dataset
            processed_data_dir: Directory containing processed data
            output_dir: Directory to save embeddings
            checkpoint_path: Path to the model checkpoint
            ecg_dataset_path: Path to the ECG dataset
            meta_dataset_path: Path to the metadata dataset
            logger: Logger instance for tracking progress (optional)
            **kwargs: Additional parameters
            
        Returns:
            An appropriately configured BaseEmbedder instance
            
        Raises:
            ValueError: If the combination of model_type and dataset_type is not supported
        """
        # Normalize model and dataset types
        model_type = model_type.lower()
        dataset_type = dataset_type.lower()
        
        # Select the appropriate embedder based on model and dataset type
        if model_type in ['ecg-fm', 'ecgfm', 'ecg_fm']:
            return ECGFMEmbedder(
                model_type=model_type,
                dataset_type=dataset_type,
                id_column=id_column,
                processed_data_dir=processed_data_dir,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                ecg_dataset_path=ecg_dataset_path,
                meta_dataset_path=meta_dataset_path,                
                logger=logger,
                **kwargs
            )
        elif any(model_type.startswith(prefix) for prefix in ['hubert-ecg', 'hubertecg', 'hubert_ecg']):
            return HuBERTECGEmbedder(
                model_type=model_type,
                dataset_type=dataset_type,
                id_column=id_column,
                processed_data_dir=processed_data_dir,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                ecg_dataset_path=ecg_dataset_path,
                meta_dataset_path=meta_dataset_path,                
                logger=logger,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

