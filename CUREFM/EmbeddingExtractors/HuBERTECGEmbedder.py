import os
import torch

from .BaseEmbedder import BaseEmbedder
from transformers import AutoModel


class HuBERTECGEmbedder(BaseEmbedder):
    """
    HuBERT-ECG specific embedder for ECG data.
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
        Initialize the HuBERT-ECG embedder with the provided parameters.
        
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
        
        # HuBERT-ECG specific parameters
        self.dataset_type = dataset_type
        
        # Dataset-specific paths
        self.embeddings_dir = os.path.join(self.embeddings_dir, 'embeddings', f'{self.dataset_type}_{self.model_type}_embeddings_vectors')
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Index file to keep track of all embeddings
        self.embeddings_index_path = os.path.join(self.embeddings_dir, 'index.csv')


    def build_model(self) -> None:
        """
        Build the HuBERT-ECG model from checkpoint.
        """
        self.log('info', f"Loading {self.model_type} model from checkpoint: {self.checkpoint_path}")
        
        self.model = AutoModel.from_pretrained(f"Edoardo-BS/{self.model_type}",
                                                cache_dir=self.checkpoint_path,
                                                trust_remote_code=True,
                                                output_hidden_states=True,
                                                output_attentions=False,
                                                attn_implementation="eager",  # needed to remove warning in output
                                              )
        # Log model architecture summary
        self.log('debug', f"Model architecture: {type(self.model).__name__}")
            
        # Move model to device and set to eval
        self.model.to(self.device)
        self.model.eval()
        
        # Calculate model size
        model_size_mb = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)  # Assuming float32 (4 bytes)
        self.log('info', f"{self.model_type} model loaded successfully. Model size: {model_size_mb:.2f} MB")
        
        # Log memory usage after model loading
        if self.logger:
            self.logger.log_memory_usage()


    # Function to embed a single ECG
    def embed_ecg(self, ecg):
        """
        Embed a single ECG signal using the HuBERT-ECG model

        Args:
            ecg: ECG signal to be embedded
        Returns:
            output: Embedded ECG signal
        """
        with torch.no_grad():
            input_tensor = ecg.to(self.device).unsqueeze(0)  # Add batch dimension
            output = self.model(input_tensor)
            embeddings = output.last_hidden_state   
            return embeddings.squeeze(0).cpu()  # Remove batch dimension, move to CPU
        