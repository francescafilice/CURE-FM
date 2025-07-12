import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, KFold


class DataSampler:
    """
    Class for sampling and preparing data for classification.
    Handles loading embedding dataframes, selecting specific labels,
    and creating balanced or unbalanced data subsets.
    """
    
    def __init__(self, config_params: Dict[str, Any], logger=None):
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
            logger: Logger instance for tracking progress
        """
        # Extract configuration parameters
        self.embedding_paths = config_params.get('embedding_paths', {})
        self.label_path = config_params.get('label_path', '')
        self.target_labels = config_params.get('target_labels', [])
        self.id_column = config_params.get('id_column')
        self.sample_percentage = config_params.get('sample_percentage', 1.0)
        self.balanced_sampling = config_params.get('balanced_sampling', True)
        self.balanced_folds = config_params.get('balanced_folds', True)
        self.test_size = config_params.get('test_size', 0.2)
        self.validation_size = config_params.get('validation_size', 0.1)
        self.n_folds = config_params.get('n_folds', 5)
        self.random_seed = config_params.get('random_seed', 42)
        
        # Logger
        self.logger = logger
        
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
        
        self.log('info', "DataSampler initialized successfully")
        self.log('debug', f"Configuration: {config_params}")
    
    
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
    

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If essential parameters are missing or have invalid values.
        """
        self.log('debug', "Validating configuration parameters...")
        
        if not self.embedding_paths:
            error_msg = "No embedding paths specified"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        if not self.label_path:
            error_msg = "Labels path not specified"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        if not self.target_labels:
            error_msg = "No target labels specified"
            self.log('error', error_msg)
            raise ValueError(error_msg)
            
        if not os.path.exists(self.label_path):
            error_msg = f"Labels file not found: {self.label_path}"
            self.log('error', error_msg)
            raise FileNotFoundError(error_msg)
        
        for pooling_method, path in self.embedding_paths.items():
            if not os.path.exists(path):
                error_msg = f"Embedding file not found for {pooling_method}: {path}"
                self.log('error', error_msg)
                raise FileNotFoundError(error_msg)
        
        if not (0.0 < self.sample_percentage <= 1.0):
            error_msg = f"Sample percentage must be between 0 and 1, found: {self.sample_percentage}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        if not (0.0 < self.test_size < 1.0):
            error_msg = f"Test size must be between 0 and 1, found: {self.test_size}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        if not (0.0 <= self.validation_size < 1.0):
            error_msg = f"Validation size must be between 0 and 1, found: {self.validation_size}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        if self.test_size + self.validation_size >= 1.0:
            error_msg = f"Sum of test_size and validation_size must be < 1, found: {self.test_size + self.validation_size}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
            
        self.log('debug', "Configuration parameters validated successfully")
    

    def _load_labels(self) -> pd.DataFrame:
        """
        Load the labels file.
        
        Returns:
            DataFrame with labels.
        """
        try:
            self.log('info', f"Loading labels from {self.label_path}...")
            labels_df = pd.read_csv(self.label_path)
            
            # Verify presence of ID column
            if self.id_column not in labels_df.columns:
                error_msg = f"ID column '{self.id_column}' not found in labels file"
                self.log('error', error_msg)
                raise ValueError(error_msg)
            
            # Verify presence of target labels
            for label in self.target_labels:
                if label not in labels_df.columns:
                    error_msg = f"Label '{label}' not found in labels file"
                    self.log('error', error_msg)
                    raise ValueError(error_msg)
            
            self.log('info', f"Labels file loaded: {labels_df.shape[0]} examples")
            
            # Log class distribution for each target label
            if self.logger:
                class_distributions = {}
                for label in self.target_labels:
                    counts = labels_df[label].value_counts().to_dict()
                    percentages = labels_df[label].value_counts(normalize=True).mul(100).to_dict()
                    class_info = {
                        str(cls): f"{count} ({percentages[cls]:.1f}%)" 
                        for cls, count in counts.items()
                    }
                    class_distributions[label] = class_info
                self.logger.log_result("Class Distribution in Labels", class_distributions)
            
            return labels_df
        
        except Exception as e:
            error_msg = f"Error loading labels file: {str(e)}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
    

    def _load_embedding(self, pooling_method: str) -> pd.DataFrame:
        """
        Load an embedding dataframe.
        
        Args:
            pooling_method: Pooling method (avg, max, min, lst)
            
        Returns:
            DataFrame with embeddings.
        """
        if pooling_method not in self.embedding_paths:
            error_msg = f"Invalid pooling method: {pooling_method}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        try:
            file_path = self.embedding_paths[pooling_method]
            self.log('info', f"Loading embeddings ({pooling_method}) from {file_path}...")
            
            if self.logger:
                self.logger.start_timer(f"load_embedding_{pooling_method}")
                
            embedding_df = pd.read_csv(file_path)
            
            # Verify presence of ID column
            if self.id_column not in embedding_df.columns:
                error_msg = f"ID column '{self.id_column}' not found in embeddings file"
                self.log('error', error_msg)
                raise ValueError(error_msg)
            
            if self.logger:
                self.logger.stop_timer(f"load_embedding_{pooling_method}")
            
            self.log('info', f"Embeddings ({pooling_method}) loaded: {embedding_df.shape[0]} examples, {embedding_df.shape[1]-1} features")
            return embedding_df
        
        except Exception as e:
            error_msg = f"Error loading embeddings file: {str(e)}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
    

    def get_embedding_dataframe(self, pooling_method: str) -> pd.DataFrame:
        """
        Get an embedding dataframe, loading it if necessary.
        
        Args:
            pooling_method: Pooling method
            
        Returns:
            DataFrame with embeddings.
        """
        if pooling_method not in self.embedding_dfs:
            self.embedding_dfs[pooling_method] = self._load_embedding(pooling_method)
        
        return self.embedding_dfs[pooling_method]
    

    def sample_data(self, pooling_method: str, target_label: str) -> pd.DataFrame:
        """
        Sample data for a specific pooling method and target label.
        
        Args:
            pooling_method: Pooling method
            target_label: Target label to use
            
        Returns:
            DataFrame with sampled data.
        """
        # Create a unique key for this combination
        key = f"{pooling_method}_{target_label}"
        
        # If data has already been sampled, return from cache
        if key in self.sampled_data:
            self.log('debug', f"Using cached sampled data for {pooling_method} and {target_label}")
            return self.sampled_data[key]
        
        self.log('info', f"Sampling data for {pooling_method} and {target_label}...")
        
        if self.logger:
            self.logger.start_timer(f"sample_data_{key}")
        
        # Get dataframes
        embedding_df = self.get_embedding_dataframe(pooling_method)
        
        # Merge embeddings and labels
        merged_df = pd.merge(
            embedding_df, 
            self.labels_df[[self.id_column, target_label]], 
            on=self.id_column,
            how='inner'
        )
        
        # Verify if there is data after merging
        if merged_df.empty:
            error_msg = f"No data available after merge for {pooling_method} and {target_label}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        self.log('info', f"Data merged for {target_label} with {pooling_method}: {merged_df.shape[0]} examples")
        
        class_counts = merged_df[target_label].value_counts().to_dict()
        class_percentages = merged_df[target_label].value_counts(normalize=True).mul(100).to_dict()
        class_info = {
            str(cls): f"{count} ({class_percentages[cls]:.1f}%)" 
            for cls, count in class_counts.items()
        }
        self.log('info', f"Class distribution: {class_info}")
        
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
            
            self.log('info', f"Data sampled: {sampled_df.shape[0]} examples ({self.sample_percentage*100:.1f}% of total)")
            
            # Log class distribution after sampling
            class_counts = sampled_df[target_label].value_counts().to_dict()
            class_percentages = sampled_df[target_label].value_counts(normalize=True).mul(100).to_dict()
            class_info = {
                str(cls): f"{count} ({class_percentages[cls]:.1f}%)" 
                for cls, count in class_counts.items()
            }
            self.log('info', f"Class distribution after sampling: {class_info}")
        else:
            sampled_df = merged_df
            self.log('info', f"Using all available data: {sampled_df.shape[0]} examples")
        
        if self.logger:
            self.logger.stop_timer(f"sample_data_{key}")
        
        # Save to cache
        self.sampled_data[key] = sampled_df
        
        return sampled_df
    

    def _balanced_sampling(self, df: pd.DataFrame, target_label: str) -> pd.DataFrame:
        """
        Perform balanced sampling for binary or multiclass classification.
        
        Args:
            df: DataFrame to sample
            target_label: Target label
            
        Returns:
            DataFrame sampled in a balanced way.
        """
        self.log('debug', f"Performing balanced sampling for {target_label}...")
        
        # Divide by class
        class_dfs = {}
        class_sizes = []
        for class_value in df[target_label].unique():
            class_df = df[df[target_label] == class_value]
            class_dfs[class_value] = class_df
            class_sizes.append(len(class_df))
            self.log('debug', f"Class {class_value}: {len(class_df)} examples")
        
        # Calculate sample size for each class
        min_class_size = min(class_sizes)

        # For multiclass, we can either use the minimum class size as reference
        # or adjust sampling to maintain class proportions while reducing overall dataset size
        if self.sample_percentage < 1.0:
            target_class_size = int(min_class_size * self.sample_percentage)
        else:
            target_class_size = min_class_size

        self.log('debug', f"Target class size: {target_class_size} examples (min class: {min_class_size})")
        
        # Sample each class separately
        sampled_dfs = []
        for class_value, class_df in class_dfs.items():
            if len(class_df) <= target_class_size:
                # If class is already smaller than or equal to target size, take all
                sampled_dfs.append(class_df)
                self.log('debug', f"Class {class_value}: keeping all {len(class_df)} examples")
            else:
                # Otherwise, sample randomly
                sampled_class_df = class_df.sample(
                    n=target_class_size, 
                    random_state=self.random_seed
                )
                sampled_dfs.append(sampled_class_df)
                self.log('debug', f"Class {class_value}: sampled {len(sampled_class_df)} from {len(class_df)} examples")
    
        # Merge sampled classes
        sampled_df = pd.concat(sampled_dfs, axis=0)
        
        # Shuffle dataframe
        sampled_df = sampled_df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        
        # Log class distribution after sampling
        class_counts = sampled_df[target_label].value_counts().to_dict()
        class_info = {str(cls): count for cls, count in class_counts.items()}
        self.log('debug', f"Balanced sampling completed: {len(sampled_df)} examples total")
        self.log('debug', f"Final class distribution: {class_info}")
        
        return sampled_df
    

    def create_train_test_split(self, pooling_method: str, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create a train-test split of the data.
        
        Args:
            pooling_method: Pooling method (avg, max, min, lst)
            target_label: Target label
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test).
        """
        self.log('info', f"Creating train-test split for {pooling_method} and {target_label}...")
        
        if self.logger:
            self.logger.start_timer(f"train_test_split_{pooling_method}_{target_label}")
        
        # Sample data
        sampled_df = self.sample_data(pooling_method, target_label)
        
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
            self.log('debug', "Performed stratified train-test split")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_seed
            )
            self.log('debug', "Performed random train-test split")
        
        self.log('info', f"Train-test split created:")
        self.log('info', f"  - Training: {X_train.shape[0]} examples")
        self.log('info', f"  - Test: {X_test.shape[0]} examples")
        
        # Log class distributions
        train_counts = y_train.value_counts().to_dict()
        train_percentages = y_train.value_counts(normalize=True).mul(100).to_dict()
        train_info = {
            str(cls): f"{count} ({train_percentages[cls]:.1f}%)" 
            for cls, count in train_counts.items()
        }
        
        test_counts = y_test.value_counts().to_dict()
        test_percentages = y_test.value_counts(normalize=True).mul(100).to_dict()
        test_info = {
            str(cls): f"{count} ({test_percentages[cls]:.1f}%)" 
            for cls, count in test_counts.items()
        }
        
        self.log('info', f"  - Class distribution in training: {train_info}")
        self.log('info', f"  - Class distribution in test: {test_info}")
        
        if self.logger:
            self.logger.stop_timer(f"train_test_split_{pooling_method}_{target_label}")
            split_info = {
                "Training samples": X_train.shape[0],
                "Test samples": X_test.shape[0],
                "Features": X_train.shape[1],
                "Training distribution": train_info,
                "Test distribution": test_info
            }
            self.logger.log_result(f"Train-Test Split ({pooling_method}, {target_label})", split_info)
        
        return X_train, X_test, y_train, y_test
    

    def create_train_val_test_split(self, pooling_method: str, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create a train-validation-test split of the data.
        
        Args:
            pooling_method: Pooling method
            target_label: Target label
            
        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        self.log('info', f"Creating train-validation-test split for {pooling_method} and {target_label}...")
        
        if self.logger:
            self.logger.start_timer(f"train_val_test_split_{pooling_method}_{target_label}")
        
        # Sample data
        sampled_df = self.sample_data(pooling_method, target_label)
        
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
            self.log('debug', "Performed stratified test split")
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_seed
            )
            self.log('debug', "Performed random test split")
        
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
            self.log('debug', "Performed stratified validation split")
        else:
            # Random split
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_size, 
                random_state=self.random_seed
            )
            self.log('debug', "Performed random validation split")
        
        self.log('info', f"Train-validation-test split created:")
        self.log('info', f"  - Training: {X_train.shape[0]} examples")
        self.log('info', f"  - Validation: {X_val.shape[0]} examples")
        self.log('info', f"  - Test: {X_test.shape[0]} examples")
        
        # Log class distributions
        train_counts = y_train.value_counts().to_dict()
        train_percentages = y_train.value_counts(normalize=True).mul(100).to_dict()
        train_info = {
            str(cls): f"{count} ({train_percentages[cls]:.1f}%)" 
            for cls, count in train_counts.items()
        }
        
        val_counts = y_val.value_counts().to_dict()
        val_percentages = y_val.value_counts(normalize=True).mul(100).to_dict()
        val_info = {
            str(cls): f"{count} ({val_percentages[cls]:.1f}%)" 
            for cls, count in val_counts.items()
        }
        
        test_counts = y_test.value_counts().to_dict()
        test_percentages = y_test.value_counts(normalize=True).mul(100).to_dict()
        test_info = {
            str(cls): f"{count} ({test_percentages[cls]:.1f}%)" 
            for cls, count in test_counts.items()
        }
        
        self.log('info', f"  - Class distribution in training: {train_info}")
        self.log('info', f"  - Class distribution in validation: {val_info}")
        self.log('info', f"  - Class distribution in test: {test_info}")
        
        if self.logger:
            self.logger.stop_timer(f"train_val_test_split_{pooling_method}_{target_label}")
            split_info = {
                "Training samples": X_train.shape[0],
                "Validation samples": X_val.shape[0],
                "Test samples": X_test.shape[0],
                "Features": X_train.shape[1],
                "Training distribution": train_info,
                "Validation distribution": val_info,
                "Test distribution": test_info
            }
            self.logger.log_result(f"Train-Val-Test Split ({pooling_method}, {target_label})", split_info)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def create_kfold_indices(self, pooling_method: str, target_label: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create indices for k-fold cross-validation.
        
        Args:
            pooling_method: Pooling method (avg, max, min, lst)
            target_label: Target label
            
        Returns:
            List of tuples (train_indices, val_indices) for each fold.
        """
        # Create a unique key for this combination
        key = f"{pooling_method}_{target_label}"
        
        # If indices have already been created, return from cache
        if key in self.fold_indices:
            self.log('debug', f"Using cached fold indices for {pooling_method} and {target_label}")
            return self.fold_indices[key]
        
        self.log('info', f"Creating {self.n_folds}-fold cross-validation indices for {pooling_method} and {target_label}...")
        
        if self.logger:
            self.logger.start_timer(f"create_kfolds_{key}")
        
        # Sample data
        sampled_df = self.sample_data(pooling_method, target_label)

        # Split into X and y
        X = sampled_df.drop(columns=[self.id_column, target_label])
        y = sampled_df[target_label]
        
        # Create fold indices
        if self.balanced_folds:
            # Stratified folds
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            fold_indices = list(kf.split(X, y))
            self.log('debug', f"Created {self.n_folds} stratified folds")
        else:
            #### AGGIUNTO
            if self.n_folds == 1: 
                ### Approach 1: 1 fold -> class appearance in train and validation set is proportional to their splits
                '''
                # Single fold: split into train and validation following percentages declared in the config file
                val_size = self.validation_size
                
                # Use stratified split to maintain class proportions
                train_indices, val_indices = train_test_split(
                    np.arange(len(X)), 
                    test_size=val_size,
                    random_state=self.random_seed,
                    stratify=y
                )
                
                fold_indices = [(train_indices, val_indices)]
                self.log('debug', f"Single fold created with train-val split: {len(train_indices)} training, {len(val_indices)} validation")
                
                # Log class distribution verification
                train_counts = y.iloc[train_indices].value_counts().to_dict()
                train_percentages = y.iloc[train_indices].value_counts(normalize=True).mul(100).to_dict()
                val_counts = y.iloc[val_indices].value_counts().to_dict()
                val_percentages = y.iloc[val_indices].value_counts(normalize=True).mul(100).to_dict()
                
                self.log('debug', f"Training class distribution: {train_counts}")
                self.log('debug', f"Validation class distribution: {val_counts}")
                self.log('debug', f"Training class percentages: {train_percentages}")
                self.log('debug', f"Validation class percentages: {val_percentages}")
                '''

                ### Approach 2: stratified shuffle split with validation_size
                # Use StratifiedShuffleSplit to generate multiple random stratified splits
                # TODO: in realtà andrebbe spostato in un altro if visto che qui non facciamo 1 fold ma siamo nell'if self.n_folds==1
                n_shuffle_splits = 10
                sss = StratifiedShuffleSplit(
                    n_splits=n_shuffle_splits, 
                    test_size=self.validation_size, 
                    random_state=self.random_seed
                )
                fold_indices = list(sss.split(X, y))
                
                self.log('debug', f"Created {n_shuffle_splits} stratified shuffle splits with validation_size={self.validation_size}")
                
                # Log class distribution verification for first split
                if fold_indices:
                    train_indices, val_indices = fold_indices[0]
                    train_counts = y.iloc[train_indices].value_counts().to_dict()
                    train_percentages = y.iloc[train_indices].value_counts(normalize=True).mul(100).to_dict()
                    val_counts = y.iloc[val_indices].value_counts().to_dict()
                    val_percentages = y.iloc[val_indices].value_counts(normalize=True).mul(100).to_dict()
                    
                    self.log('debug', f"Sample split - Training class distribution: {train_counts}")
                    self.log('debug', f"Sample split - Validation class distribution: {val_counts}")
                    self.log('debug', f"Sample split - Training class percentages: {train_percentages}")
                    self.log('debug', f"Sample split - Validation class percentages: {val_percentages}")
            #### FINE AGGIUNTO
            else:
                # Random folds
                kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
                fold_indices = list(kf.split(X))
                self.log('debug', f"Created {self.n_folds} random folds")
        
        # Save to cache
        self.fold_indices[key] = fold_indices
        
        self.log('info', f"{self.n_folds}-fold cross-validation indices created for {pooling_method} and {target_label}")
        
        # Log fold sizes
        fold_info = {}
        for i, (train_indices, val_indices) in enumerate(fold_indices):
            fold_info[f"Fold {i+1}"] = f"{len(train_indices)} training, {len(val_indices)} validation"
            self.log('debug', f"  - Fold {i+1}: {len(train_indices)} training, {len(val_indices)} validation")
            
            # Log class distribution in each fold if using stratified sampling
            if self.balanced_folds:
                train_counts = y.iloc[train_indices].value_counts().to_dict()
                train_percentages = y.iloc[train_indices].value_counts(normalize=True).mul(100).to_dict()
                train_info = {
                    str(cls): f"{count} ({train_percentages[cls]:.1f}%)" 
                    for cls, count in train_counts.items()
                }
                
                val_counts = y.iloc[val_indices].value_counts().to_dict()
                val_percentages = y.iloc[val_indices].value_counts(normalize=True).mul(100).to_dict()
                val_info = {
                    str(cls): f"{count} ({val_percentages[cls]:.1f}%)" 
                    for cls, count in val_counts.items()
                }
                
                self.log('debug', f"    - Training distribution: {train_info}")
                self.log('debug', f"    - Validation distribution: {val_info}")
        
        if self.logger:
            self.logger.stop_timer(f"create_kfolds_{key}")
            self.logger.log_result(f"K-Fold Indices ({pooling_method}, {target_label})", fold_info)
        
        return fold_indices
    

    def get_fold_data(self, pooling_method: str, target_label: str, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get data for a specific fold.
        
        Args:
            pooling_method: Pooling method (avg, max, min, lst)
            target_label: Target label
            fold_idx: Fold index (0-based)
            
        Returns:
            Tuple (X_train, X_val, y_train, y_val) for the specified fold.
        """
        self.log('info', f"Getting data for fold {fold_idx+1} for {pooling_method} and {target_label}...")
        
        # Get fold indices
        fold_indices = self.create_kfold_indices(pooling_method, target_label)
        
        # Verify that fold index is valid
        if fold_idx < 0 or fold_idx >= len(fold_indices):
            error_msg = f"Invalid fold index: {fold_idx}. Must be between 0 and {len(fold_indices)-1}"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        # Get training and validation indices for this fold
        train_indices, val_indices = fold_indices[fold_idx]
        
        # Sample data
        sampled_df = self.sample_data(pooling_method, target_label)
        
        # Split into X and y
        X = sampled_df.drop(columns=[self.id_column, target_label])
        y = sampled_df[target_label]
        
        # Create training and validation dataframes
        X_train = X.iloc[train_indices]
        X_val = X.iloc[val_indices]
        y_train = y.iloc[train_indices]
        y_val = y.iloc[val_indices]
        
        self.log('info', f"Data for fold {fold_idx+1}: {len(X_train)} training, {len(X_val)} validation samples")
        
        # Log class distribution in this fold
        train_counts = y_train.value_counts().to_dict()
        train_percentages = y_train.value_counts(normalize=True).mul(100).to_dict()
        train_info = {
            str(cls): f"{count} ({train_percentages[cls]:.1f}%)" 
            for cls, count in train_counts.items()
        }
        
        val_counts = y_val.value_counts().to_dict()
        val_percentages = y_val.value_counts(normalize=True).mul(100).to_dict()
        val_info = {
            str(cls): f"{count} ({val_percentages[cls]:.1f}%)" 
            for cls, count in val_counts.items()
        }
        
        self.log('debug', f"  - Training distribution: {train_info}")
        self.log('debug', f"  - Validation distribution: {val_info}")
        
        return X_train, X_val, y_train, y_val
    

    def get_all_pooling_methods(self) -> List[str]:
        """
        Get all available pooling methods.
        
        Returns:
            List of available pooling methods.
        """
        methods = list(self.embedding_paths.keys())
        self.log('debug', f"Available pooling methods: {methods}")
        return methods
    

    def get_all_target_labels(self) -> List[str]:
        """
        Get all available target labels.
        
        Returns:
            List of available target labels.
        """
        self.log('debug', f"Available target labels: {self.target_labels}")
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
            error_msg = f"Label '{target_label}' not found in labels file"
            self.log('error', error_msg)
            raise ValueError(error_msg)
        
        distribution = self.labels_df[target_label].value_counts().to_dict()
        self.log('debug', f"Data distribution for {target_label}: {distribution}")
        return distribution


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
        
        info = {
            'count': len(feature_cols),
            'names': feature_cols
        }
        
        self.log('debug', f"Feature information for {pooling_method}: {len(feature_cols)} features")
        return info