import os
import torch
from .BaseEmbedder import BaseEmbedder
from .tools.ECGFMTools.wave2vec2_cmsc_custom import Wav2Vec2CMSCConfigCustom, Wav2Vec2CMSCModelCustom


class ECGFMEmbedder(BaseEmbedder):
    """
    ECG-FM specific embedder for ECG data.
    Implements the methods of the BaseEmbedder abstract class.
    """
    
    def __init__(self,
                 model_type: str,
                 dataset_type: str,
                 id_column: str,
                 processed_data_dir: str,
                 output_dir: str,
                 checkpoint_path: str,
                 ecg_dataset_path: str,
                 meta_dataset_path: str,
                 **kwargs):
        """
        Initialize the ECG-FM embedder with the provided parameters.
        
        Args:
            model_type: Type of model (e.g., "HuBERT-ECG-small")
            dataset_type: Name of the dataset
            id_column: Column name for the ECG ID 
            processed_data_dir: Directory containing processed data
            output_dir: Directory to save embeddings
            checkpoint_path: Path to the model checkpoint
            ecg_dataset_path: Path to the ECG dataset
            meta_dataset_path: Path to the metadata dataset
            **kwargs: Additional parameters
        """
        super().__init__(
            model_type=model_type,
            dataset_type=dataset_type,
            id_column=id_column,
            processed_data_dir=processed_data_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            ecg_dataset_path=ecg_dataset_path,
            meta_dataset_path=meta_dataset_path,
            **kwargs
        )
        
        # ECG-FM specific parameters
        self.dataset_type = dataset_type
        
        # Dataset-specific paths
        self.embeddings_dir = os.path.join(self.output_dir, 'embeddings', f'{self.dataset_type}_ecgfm_embeddings_vectors')
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Index file to keep track of all embeddings
        self.embeddings_index_path = os.path.join(self.embeddings_dir, 'index.csv')
        
        # Log ECG-FM specific parameters
        self.log('debug', f"Dataset name: {self.dataset_type}")
        self.log('debug', f"Embeddings directory: {self.embeddings_dir}")
    
    
    def build_model(self) -> None:
        """
        Build the ECG-FM model from checkpoint.
        """
        self.log('info', f"Loading ECG-FM model from checkpoint: {self.checkpoint_path}")
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create ECG-FM model
            config = Wav2Vec2CMSCConfigCustom(quantize_targets=False)
            self.log('debug', f"Model configuration: {str(config)}")
            
            self.model = Wav2Vec2CMSCModelCustom(config)
            
            # Log model architecture summary
            self.log('debug', f"Model architecture: {type(self.model).__name__}")
            
            # Load checkpoint weights
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Calculate model size
            model_size_mb = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)  # Assuming float32 (4 bytes)
            self.log('info', f"ECG-FM model loaded successfully. Model size: {model_size_mb:.2f} MB")
            
            # Log memory usage after model loading
            if self.logger:
                self.logger.log_memory_usage()
                
        except Exception as e:
            error_msg = f"Failed to load ECG-FM model: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
    

    # Function to embed a single ECG
    def embed_ecg(self, ecg):
        """
        Embed a single ECG signal using the ECG-FM model

        Args:
            ecg: ECG signal to be embedded
        Returns:
            output: Embedded ECG signal
        """
        with torch.no_grad():
            input_tensor = ecg.to(self.device).unsqueeze(0)  # Add batch dimension
            output = self.model(source=input_tensor)['x']  # Shape: [1, 156, 768]
            return output.squeeze(0).cpu()  # Remove batch dimension, move to CPU
            