from abc import ABC, abstractmethod
import os
import torch
import pandas as pd
from typing import List
from tqdm import tqdm


class BaseEmbedder(ABC):
    """
    Abstract base class for all ECG embedding extractors.
    Defines the common interface that all embedders must implement.
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
                 logger=None,
                 **kwargs):
        """
        Initialize the embedder with the provided parameters.
        
        Args:
            model_type: Type of model (e.g., "ECG-FM")
            dataset_type: Type of dataset (e.g., "code15")
            id_column: Column name for the exam ID
            processed_data_dir: Directory for processed data
            output_dir: Directory for output embeddings
            checkpoint_path: Path to the model checkpoint
            ecg_dataset_path: Path where the processed ECG dataset is located
            meta_dataset_path: Path to save metadata
            logger: Logger object for logging messages
            **kwargs: Additional parameters
        """
        # Base paths
        self.package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Input data paths
        self.processed_data_dir = processed_data_dir
        self.ecg_dataset_path = ecg_dataset_path
        self.meta_dataset_path = meta_dataset_path
        
        # Output paths
        self.output_dir = output_dir
        self.embeddings_dir = output_dir
        self.pooled_dir = os.path.join(output_dir, 'pooled')
        
        # model-dataset specific parameters
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.id_column = id_column
        
        # Logger
        self.logger = logger
        
        # Other parameters
        self.kwargs = kwargs
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.pooled_dir, exist_ok=True)
        
        # Load model
        self.model = None
        
        # Log initialization
        self.log('info', f"Initialized {self.__class__.__name__}")
        self.log('debug', f"Processed data directory: {self.processed_data_dir}")
        self.log('debug', f"ECG dataset path: {self.ecg_dataset_path}")
        self.log('debug', f"Metadata dataset path: {self.meta_dataset_path}")
        self.log('debug', f"Output directory: {self.output_dir}")
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
        Build the foundation model for embedding extraction.
        """
        pass


    @abstractmethod
    def embed_ecg(self, ecg):
        """
        Embed a single ECG tensor to generate its embedding.
        
        Args:
            ecg: The ECG tensor to embed
            
        Returns:
            The embedding tensor
        """
        pass

    
    def extract_embeddings(self) -> None:
        """
        Extract embeddings from the ECG data using the model.
        Each embedding is saved as a separate torch tensor file with the exam_id as filename.
        """
        # Check if model is loaded
        if self.model is None:
            self.log('error', "Model is not loaded. Call build_model() first.")
            raise RuntimeError("Model is not loaded. Call build_model() first.")
        
        # Check if index file already exists
        if os.path.exists(self.embeddings_index_path):
            self.log('info', f"Embedding index file already exists at {self.embeddings_index_path}")
            index_df = pd.read_csv(self.embeddings_index_path)
            existing_exam_ids = set(index_df[self.id_column].astype(str).tolist())
            self.log('info', f"Found {len(existing_exam_ids)} existing embeddings.")
        else:
            existing_exam_ids = set()
            self.log('info', "No existing embeddings found. Starting from scratch.")
        
        self.log('info', f"Extracting embeddings for {self.dataset_type} dataset...")
        
        # Load ECG dataset
        try:
            ecg_df = pd.read_pickle(self.ecg_dataset_path)
            self.log('info', f"Loaded ECG dataset with {len(ecg_df)} records.")
        except Exception as e:
            error_msg = f"Failed to load ECG dataset from {self.ecg_dataset_path}: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
        
        # Extract embeddings for each ECG
        index_entries = []
        
        # Create progress bar with consistent output
        total_ecgs = len(ecg_df)
        processed_count = 0
        skipped_count = 0
        success_count = 0
        error_count = 0
        
        # Log start of extraction
        self.log('info', f"Starting extraction for {total_ecgs} ECGs...")
        
        for _, row in tqdm(ecg_df.iterrows(), total=total_ecgs, desc=f"Extracting {self.__class__.__name__} embeddings"):
            processed_count += 1
            
            try:
                ecg_tensor = row['ecg']
                exam_id = str(row[self.id_column])
                
                # Skip if already processed
                if exam_id in existing_exam_ids:
                    self.log('debug', f"Skipping already processed exam_id: {exam_id}")
                    skipped_count += 1
                    continue
                
                # Extract embeddings using the model-specific embed_ecg method
                embedding = self.embed_ecg(ecg_tensor)
                
                # Save tensor with exam_id as filename
                embedding_path = os.path.join(self.embeddings_dir, f"{exam_id}.pt")
                torch.save(embedding, embedding_path)
                
                # Add to index
                index_entries.append({
                    f'{self.id_column}': exam_id,
                    'embedding_path': embedding_path,
                    'shape': str(list(embedding.shape))
                })
                
                success_count += 1
                
            except Exception as e:
                error_msg = f"Error processing ECG for exam_id {row.get(self.id_column, 'unknown')}: {str(e)}"
                self.log('error', error_msg)
                error_count += 1
        
        # Create or update index DataFrame
        if index_entries:
            new_index_df = pd.DataFrame(index_entries)
            if os.path.exists(self.embeddings_index_path):
                # Merge with existing index
                existing_index_df = pd.read_csv(self.embeddings_index_path)
                combined_index_df = pd.concat([existing_index_df, new_index_df], ignore_index=True)
                combined_index_df.to_csv(self.embeddings_index_path, index=False)
                self.log('info', f"Updated index with {len(index_entries)} new entries, total {len(combined_index_df)} embeddings")
            else:
                # Create new index
                new_index_df.to_csv(self.embeddings_index_path, index=False)
                self.log('info', f"Created new index with {len(index_entries)} embeddings")
            
            self.log('info', f"Summary: {processed_count} ECGs processed, {success_count} embeddings extracted, "
                          f"{skipped_count} skipped, {error_count} errors")
        else:
            self.log('info', "No new embeddings to extract.")
        
        # Log memory usage after extraction
        if self.logger:
            self.logger.log_memory_usage()

    
    def pool_embeddings(self, pool_methods: List[str]) -> None:
        """
        Pool the embeddings using all specified methods at once.
        Results are saved as CSV files.
        
        Args:
            pool_methods: List of pooling methods to use (avg, max, min, lst)
        """

        # Define here all the pooling functions
        def max_pooling(embedding):
            return embedding.max(dim=0).values
            
        def lst_pooling(embedding):
            return embedding[-1]
        
        # Map method names to pooling functions
        pooling_functions = {
            'max': max_pooling,
            'lst': lst_pooling
        }
        
        # Check if index file exists
        if not os.path.exists(self.embeddings_index_path):
            error_msg = f"Embedding index file not found at {self.embeddings_index_path}. Run extract_embeddings first."
            self.log('error', error_msg)
            raise FileNotFoundError(error_msg)
        
        self.log('info', f"Pooling embeddings using methods: {', '.join(pool_methods)}...")
        
        # Load embeddings index
        try:
            index_df = pd.read_csv(self.embeddings_index_path)
            self.log('info', f"Loaded embedding index with {len(index_df)} entries.")
        except Exception as e:
            error_msg = f"Failed to load embedding index from {self.embeddings_index_path}: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
        
        # Create placeholders for pooled data
        pooled_data = {method: [] for method in pool_methods}
        
        # Track processing statistics
        total_count = len(index_df)
        processed_count = 0
        error_count = 0
        
        # Identify if we've already processed some embeddings (to avoid duplicate work)
        existing_pooled_files = {}
        existing_exam_ids = {method: set() for method in pool_methods}
        
        for method in pool_methods:
            pooled_path = os.path.join(self.pooled_dir, f'{self.dataset_type}_{self.model_type}_{method}_pooled.csv')
            existing_pooled_files[method] = pooled_path
            
            if os.path.exists(pooled_path):
                # Remove the existing pooled file and start anew
                self.log('info', f"Removing existing pooled file: {pooled_path}")
                os.remove(pooled_path)
                existing_exam_ids[method] = set() 
            
        
        # Process all ECGs and apply all pooling methods at once
        self.log('info', f"Starting pooling for {total_count} embeddings...")
        
        for _, row in tqdm(index_df.iterrows(), total=total_count, desc="Pooling embeddings"):
            processed_count += 1
            
            exam_id = row[self.id_column]
            embedding_path = row['embedding_path']
            
            # Load embedding tensor
            try:
                embedding = torch.load(embedding_path)
                
                # Apply each pooling method
                for method in pool_methods:
                    # Skip if already processed for this method
                    if exam_id in existing_exam_ids.get(method, set()):
                        continue
                        
                    if method in pooling_functions:
                        pooled = pooling_functions[method](embedding)
                        pooled_np = pooled.numpy()
                        
                        # Create row with exam_id and features
                        row_dict = {self.id_column: exam_id}
                        row_dict.update({f'feature_{i}': val for i, val in enumerate(pooled_np)})
                        pooled_data[method].append(row_dict)
                    else:
                        self.log('warning', f"Unknown pooling method '{method}'. Skipping.")
            except Exception as e:
                error_msg = f"Error pooling embedding at {embedding_path}: {str(e)}"
                self.log('error', error_msg)
                error_count += 1
                continue
        
        for method in pool_methods:
            if method in pooling_functions and pooled_data[method]:
                pooled_path = os.path.join(self.pooled_dir, f'{self.dataset_type}_{self.model_type}_{method}_pooled.csv')
                
                new_pooled_df = pd.DataFrame(pooled_data[method])
                
                # If file exists, append to it
                if os.path.exists(pooled_path):
                    try:
                        existing_df = pd.read_csv(pooled_path)
                        combined_df = pd.concat([existing_df, new_pooled_df], ignore_index=True)
                        combined_df.to_csv(pooled_path, index=False)
                        self.log('info', f"Updated {method} pooled features at {pooled_path}: "
                                      f"{len(existing_df)} existing + {len(new_pooled_df)} new = {len(combined_df)} total")
                    except Exception as e:
                        # If there's an error, create a backup and write a new file
                        backup_path = f"{pooled_path}.bak"
                        self.log('warning', f"Error appending to existing file. Creating backup at {backup_path}")
                        os.rename(pooled_path, backup_path)
                        new_pooled_df.to_csv(pooled_path, index=False)
                        self.log('info', f"Saved {len(new_pooled_df)} {method} pooled features to {pooled_path}")
                else:
                    # Create new file
                    new_pooled_df.to_csv(pooled_path, index=False)
                    self.log('info', f"Saved {len(new_pooled_df)} {method} pooled features to {pooled_path}")
            elif method in pooling_functions:
                self.log('info', f"No new {method} pooled features to save.")
        
        self.log('info', f"Completed pooling of embeddings with {len(pool_methods)} methods. "
                      f"Processed {processed_count} embeddings with {error_count} errors.")
        
        # Log memory usage after pooling
        if self.logger:
            self.logger.log_memory_usage()
            
        # Log feature dimensions for each pooling method
        feature_dimensions = {}
        for method in pool_methods:
            if method in pooling_functions:
                pooled_path = os.path.join(self.pooled_dir, f'{self.dataset_type}_{self.model_type}_{method}_pooled.csv')
                if os.path.exists(pooled_path):
                    try:
                        df = pd.read_csv(pooled_path)
                        feature_cols = [col for col in df.columns if col.startswith('feature_')]
                        feature_dimensions[method] = len(feature_cols)
                    except Exception:
                        feature_dimensions[method] = "Error reading file"
                else:
                    feature_dimensions[method] = "File not created"
        
        if self.logger:
            self.logger.log_result("Feature Dimensions", feature_dimensions)

    
    def _log_start(self, task: str, message: str) -> None:
        self.log('info', message)
        if self.logger:
            self.logger.start_timer(task)

    
    def _log_end(self, task: str) -> None:
        if self.logger:
            self.logger.stop_timer(task)
        

    def process_all(self, pool_methods: List[str]) -> None:
        """
        Process all steps: build model, extract embeddings, and pool embeddings.
        
        Args:
            pool_methods: List of pooling methods to use
        """
        self.log('info', "Starting embedding extraction process...")
        
        # 1. Build model
        self._log_start("build_model", "Building model...")
        self.build_model()
        self._log_end("build_model")
        
        # 2. Extract embeddings
        self._log_start("extract_embeddings", "Extracting embeddings...")
        self.extract_embeddings()
        self._log_end("extract_embeddings")
        
        # 3. Pool embeddings - all methods at once
        self._log_start("pool_embeddings", f"Pooling embeddings using methods: {', '.join(pool_methods)}...")
        self.pool_embeddings(pool_methods)
        self._log_end("pool_embeddings")
        
        self.log('info', "Embedding extraction completed successfully.")
        
        # Log memory usage
        if self.logger:
            self.logger.log_memory_usage()
