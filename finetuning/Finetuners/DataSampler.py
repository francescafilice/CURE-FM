import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


class DataSampler:
    """
    Class for sampling and preparing data for classification.
    Handles loading embedding dataframes, selecting specific labels,
    and creating balanced or unbalanced data subsets.
    """
    
    def __init__(self, config_params: Dict[str, Any]):
        """
        Initialize the data sampler.
        
        Args:
            config_params: Configuration dictionary with the following parameters:
                - embedding_paths: Paths to embedding dataframes
                - label_path: Path to the labels file
                - target_labels: List of target labels to use
                - id_column: Name of the column with identifiers
                - sample_percentage: Percentage of data to sample
                - balanced_sampling: Whether to perform balanced sampling
                - balanced_folds: Whether to create balanced folds
                - test_size: Proportion of the test set
                - n_folds: Number of folds for cross-validation
                - random_seed: Seed for reproducibility
        """
        # Extract configuration parameters
        self.ecg_path = config_params.get('ecg_path', '')
        self.label_path = config_params.get('label_path', '')
        self.target_labels = config_params.get('target_labels', [])
        self.id_column = config_params.get('id_column', 'exam_id')
        self.sample_percentage = config_params.get('sample_percentage', 1.0)
        self.balanced_sampling = config_params.get('balanced_sampling', True)
        self.balanced_folds = config_params.get('balanced_folds', True)
        self.test_size = config_params.get('test_size', 0.2)
        self.validation_size = config_params.get('validation_size', 0.1)
        self.n_folds = config_params.get('n_folds', 5)
        self.random_seed = config_params.get('random_seed', 42)
        
        # Validate parameters
        self._validate_config()
        
        # Load labels
        self.labels_df = self._load_labels()
        
        # Dictionary to store embedding dataframes
        self.embedding_dfs = {}
        
        # Cache for sampled data
        self.sampled_data = {}
        
        # Cache for folds
        self.fold_indices = {}
        
        print("DataSampler initialized successfully")
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If essential parameters are missing or have invalid values.
        """
        if not self.ecg_path:
            raise ValueError("ECG path not specified")
        
        if not self.label_path:
            raise ValueError("Labels path not specified")
        
        if not self.target_labels:
            raise ValueError("No target labels specified")
            
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Labels file not found: {self.label_path}")
        
        if not (0.0 < self.sample_percentage <= 1.0):
            raise ValueError(f"Sample percentage must be between 0 and 1, found: {self.sample_percentage}")
        
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"Test size must be between 0 and 1, found: {self.test_size}")
        
        if not (0.0 <= self.validation_size < 1.0):
            raise ValueError(f"Validation size must be between 0 and 1, found: {self.validation_size}")
        
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError(f"Sum of test_size and validation_size must be < 1, found: {self.test_size + self.validation_size}")
    
    def _load_labels(self) -> pd.DataFrame:
        """
        Load the labels file.
        
        Returns:
            DataFrame with labels.
        """
        try:
            labels_df = pd.read_csv(self.label_path)
            
            # Verify presence of ID column
            if self.id_column not in labels_df.columns:
                raise ValueError(f"ID column '{self.id_column}' not found in labels file")
            
            # Verify presence of target labels
            for label in self.target_labels:
                if label not in labels_df.columns:
                    raise ValueError(f"Label '{label}' not found in labels file")
            
            print(f"Labels file loaded: {labels_df.shape[0]} examples")
            return labels_df
        
        except Exception as e:
            raise ValueError(f"Error loading labels file: {str(e)}")
    
    
    def sample_data(self, target_label: str) -> pd.DataFrame:
        """
        Sample data from ECG embeddings contained in a pickle file.
        
        Args:
            target_label: Target label to use
                
        Returns:
            DataFrame with sampled data.
        """
        # Create a unique key for this combination
        key = f"ecg_{target_label}"
        
        # If data has already been sampled, return from cache
        if key in self.sampled_data:
            return self.sampled_data[key]
        
        # Load pickle file with ECG embeddings
        try:
            import pickle
            with open(self.ecg_path, 'rb') as f:
                ecg_embeddings = pickle.load(f)
            
            print(f"ECG embedding file loaded from {self.ecg_path}")
            
            # Convert embeddings to DataFrame
            # Assuming ecg_embeddings is a dictionary with ID keys and embedding values
            # or an already formatted DataFrame
            if isinstance(ecg_embeddings, dict):
                embedding_data = []
                for id, embedding in ecg_embeddings.items():
                    row = {self.id_column: id}
                    # Add each embedding value as a column
                    if isinstance(embedding, (list, np.ndarray)):
                        for i, val in enumerate(embedding):
                            row[f"feature_{i}"] = val
                    elif isinstance(embedding, dict):
                        for key, val in embedding.items():
                            row[key] = val
                    embedding_data.append(row)
                embedding_df = pd.DataFrame(embedding_data)
            elif isinstance(ecg_embeddings, pd.DataFrame):
                embedding_df = ecg_embeddings
            else:
                raise ValueError(f"Unsupported ECG embedding format: {type(ecg_embeddings)}")
            
        except Exception as e:
            raise ValueError(f"Error loading ECG embedding file: {str(e)}")
        
        # Merge embeddings and labels
        merged_df = pd.merge(
            embedding_df, 
            self.labels_df[[self.id_column, target_label]], 
            on=self.id_column,
            how='inner'
        )
        
        # Verify if there is data after merging
        if merged_df.empty:
            raise ValueError(f"No data available after merge for ECG embeddings and {target_label}")
        
        print(f"Data merged for {target_label}: {merged_df.shape[0]} examples")
        print(f"Class distribution: {merged_df[target_label].value_counts().to_dict()}")
        
        # Data sampling
        if self.sample_percentage < 1.0:
            if self.balanced_sampling:
                # Stratified sampling
                sampled_df = self._balanced_sampling(merged_df, target_label)
            else:
                # Random sampling
                sampled_df = merged_df.sample(
                    frac=self.sample_percentage, 
                    random_state=self.random_seed
                )
            
            print(f"Data sampled: {sampled_df.shape[0]} examples ({self.sample_percentage*100:.1f}% of total)")
            print(f"Class distribution after sampling: {sampled_df[target_label].value_counts().to_dict()}")
        else:
            sampled_df = merged_df
            print(f"Using all available data: {sampled_df.shape[0]} examples")
        
        # Save to cache
        self.sampled_data[key] = sampled_df
        
        return sampled_df
    
    def _balanced_sampling(self, df: pd.DataFrame, target_label: str) -> pd.DataFrame:
        """
        Perform balanced sampling.
        
        Args:
            df: DataFrame to sample
            target_label: Target label
            
        Returns:
            DataFrame sampled in a balanced way.
        """
        # Divide by class
        class_dfs = {}
        for class_value in df[target_label].unique():
            class_df = df[df[target_label] == class_value]
            class_dfs[class_value] = class_df
        
        # Calculate sample size for each class
        min_class_size = min(len(df) for df in class_dfs.values())
        target_class_size = int(min_class_size * self.sample_percentage)
        
        # Sample each class separately
        sampled_dfs = []
        for class_value, class_df in class_dfs.items():
            if len(class_df) <= target_class_size:
                # If class is already smaller than or equal to target size, take all
                sampled_dfs.append(class_df)
            else:
                # Otherwise, sample randomly
                sampled_dfs.append(class_df.sample(
                    n=target_class_size, 
                    random_state=self.random_seed
                ))
        
        # Merge sampled classes
        sampled_df = pd.concat(sampled_dfs, axis=0)
        
        # Shuffle dataframe
        sampled_df = sampled_df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        
        return sampled_df
    
    def create_train_test_split(self, pooling_method: str, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create a train-test split of the data.
        
        Args:
            pooling_method: Pooling method
            target_label: Target label
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test).
        """
        # Sample data
        sampled_df = self.sample_data(target_label)
        
        # Split into X and y
        X = sampled_df.drop(columns=[self.id_column, target_label])
        y = sampled_df[target_label]
        
        # Train-test split
        if self.balanced_sampling:
            # Stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_seed,
                stratify=y
            )
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_seed
            )
        
        print(f"Train-test split created:")
        print(f"  - Training: {X_train.shape[0]} examples")
        print(f"  - Test: {X_test.shape[0]} examples")
        print(f"  - Class distribution in training: {y_train.value_counts().to_dict()}")
        print(f"  - Class distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def create_train_val_test_split(self, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create a train-validation-test split of the data.
        
        Args:
            target_label: Target label
            
        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # Sample data
        sampled_df = self.sample_data(target_label)
        
        # Split into X and y
        X = sampled_df.drop(columns=[self.id_column, target_label])
        y = sampled_df[target_label]
        
        # First split: separating test set
        if self.balanced_sampling:
            # Stratified split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_seed,
                stratify=y
            )
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_seed
            )
        
        # Second split: separating validation set
        val_size = self.validation_size / (1 - self.test_size)  # Adjust validation set size
        
        if self.balanced_sampling:
            # Stratified split
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_size, 
                random_state=self.random_seed,
                stratify=y_temp
            )
        else:
            # Random split
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_size, 
                random_state=self.random_seed
            )
        
        print(f"Train-validation-test split created:")
        print(f"  - Training: {X_train.shape[0]} examples")
        print(f"  - Validation: {X_val.shape[0]} examples")
        print(f"  - Test: {X_test.shape[0]} examples")
        print(f"  - Class distribution in training: {y_train.value_counts().to_dict()}")
        print(f"  - Class distribution in validation: {y_val.value_counts().to_dict()}")
        print(f"  - Class distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_kfold_indices(self, pooling_method: str, target_label: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create indices for k-fold cross-validation.
        
        Args:
            pooling_method: Pooling method
            target_label: Target label
            
        Returns:
            List of tuples (train_indices, val_indices) for each fold.
        """
        # Create a unique key for this combination
        key = f"{pooling_method}_{target_label}"
        
        # If indices have already been created, return from cache
        if key in self.fold_indices:
            return self.fold_indices[key]
        
        # Sample data
        sampled_df = self.sample_data(target_label)
        
        # Split into X and y
        X = sampled_df.drop(columns=[self.id_column, target_label])
        y = sampled_df[target_label]
        
        # Create fold indices
        if self.balanced_folds:
            # Stratified folds
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            fold_indices = list(kf.split(X, y))
        else:
            # Random folds
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            fold_indices = list(kf.split(X))
        
        # Save to cache
        self.fold_indices[key] = fold_indices
        
        print(f"{self.n_folds}-fold cross-validation indices created for {pooling_method} and {target_label}")
        for i, (train_indices, val_indices) in enumerate(fold_indices):
            print(f"  - Fold {i+1}: {len(train_indices)} training, {len(val_indices)} validation")
        
        return fold_indices
    
    def get_fold_data(self, pooling_method: str, target_label: str, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get data for a specific fold.
        
        Args:
            pooling_method: Pooling method
            target_label: Target label
            fold_idx: Fold index (0-based)
            
        Returns:
            Tuple (X_train, X_val, y_train, y_val) for the specified fold.
        """
        # Get fold indices
        fold_indices = self.create_kfold_indices(pooling_method, target_label)
        
        # Verify that fold index is valid
        if fold_idx < 0 or fold_idx >= len(fold_indices):
            raise ValueError(f"Invalid fold index: {fold_idx}. Must be between 0 and {len(fold_indices)-1}")
        
        # Get training and validation indices for this fold
        train_indices, val_indices = fold_indices[fold_idx]
        
        # Sample data
        sampled_df = self.sample_data(target_label)
        
        # Split into X and y
        X = sampled_df.drop(columns=[self.id_column, target_label])
        y = sampled_df[target_label]
        
        # Create training and validation dataframes
        X_train = X.iloc[train_indices]
        X_val = X.iloc[val_indices]
        y_train = y.iloc[train_indices]
        y_val = y.iloc[val_indices]
        
        return X_train, X_val, y_train, y_val
    
    def get_all_pooling_methods(self) -> List[str]:
        """
        Get all available pooling methods.
        
        Returns:
            List of available pooling methods.
        """
        return list(self.embedding_paths.keys())
    
    def get_all_target_labels(self) -> List[str]:
        """
        Get all available target labels.
        
        Returns:
            List of available target labels.
        """
        return self.target_labels
    
    def get_data_distribution(self, target_label: str) -> Dict[Any, int]:
        """
        Get data distribution for a target label.
        
        Args:
            target_label: Target label
            
        Returns:
            Dictionary with class distribution.
        """
        if target_label not in self.labels_df.columns:
            raise ValueError(f"Label '{target_label}' not found in labels file")
        
        return self.labels_df[target_label].value_counts().to_dict()
    
    def get_features_info(self, pooling_method: str) -> Dict[str, Any]:
        """
        Get information about features for a pooling method.
        
        Args:
            pooling_method: Pooling method
            
        Returns:
            Dictionary with feature information.
        """
        embedding_df = self.get_embedding_dataframe(pooling_method)
        
        # Exclude ID column
        feature_cols = [col for col in embedding_df.columns if col != self.id_column]
        
        return {
            'count': len(feature_cols),
            'names': feature_cols
        }