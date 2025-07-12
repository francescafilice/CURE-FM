import os
import shutil
import subprocess
import pandas as pd
import scipy.io
import torch
from typing import List, Optional, Tuple

from .BasePreprocessor import BasePreprocessor


class Code15ECGFMPreprocessor(BasePreprocessor):
    """
    Specific preprocessor for the Code-15 ECG dataset with ECG-FM model.
    Implements the methods of the BasePreprocessor base class.
    """
    
    def __init__(self,
                 raw_data_path: str,
                 processed_data_dir: str,
                 ecg_dataset_path: str,
                 meta_dataset_path: str,
                 labels: Optional[List[str]] = None,
                 primary_label: str = "normal_ecg",
                 num_parts: int = 2,
                 logger=None,
                 **kwargs):
        """
        Initialize the preprocessor with the provided parameters.
        
        Args:
            raw_data_path: Path to raw data
            processed_data_dir: Directory to save processed data
            ecg_dataset_path: Path to save the ECG dataset
            meta_dataset_path: Path to save metadata
            labels: List of labels to extract (if None, will discover from data)
            primary_label: Primary label for classification
            num_parts: Number of HDF5 parts to process
            logger: Logger instance for tracking progress
            **kwargs: Additional parameters
        """
        super().__init__(
            raw_data_path=raw_data_path,
            processed_data_dir=processed_data_dir,
            ecg_dataset_path=ecg_dataset_path,
            meta_dataset_path=meta_dataset_path,
            logger=logger,
            labels=labels,
            primary_label=primary_label,
            num_parts=num_parts,
            **kwargs
        )
        
        # Specific parameters for Code-15
        self.labels = labels
        self.primary_label = primary_label
        self.num_parts = num_parts
        
        # Processor script paths
        self.signals_script = os.path.join(self.package_dir, 'Preprocessors', 'preprocess_cod15_ecgfm', 'code_15_signals.py')
        self.splits_script = os.path.join(self.package_dir, 'Preprocessors', 'preprocess_cod15_ecgfm', 'splits.py')
        self.labels_script = os.path.join(self.package_dir, 'Preprocessors', 'preprocess_cod15_ecgfm', 'code_15_labels.py')
        
        # Additional logging for Code15 specific parameters
        self.log('debug', f"Labels to extract: {self.labels}")
        self.log('debug', f"Primary label: {self.primary_label}")
        self.log('debug', f"Number of HDF5 parts: {self.num_parts}")

    
    def preprocess(self) -> None:
        """
        Execute complete preprocessing of the Code-15 dataset.
        """
        self.log('info', "Starting complete preprocessing of the Code-15 dataset")
        
        # Remove existing datasets if present
        if os.path.exists(self.ecg_dataset_path):
            self.log('info', f"Removing existing ECG dataset: {self.ecg_dataset_path}")
            os.remove(self.ecg_dataset_path)
        if os.path.exists(self.meta_dataset_path):
            self.log('info', f"Removing existing metadata dataset: {self.meta_dataset_path}")
            os.remove(self.meta_dataset_path)
        
        # Process each HDF5 part
        for i in range(self.num_parts):  
            self.log('info', f"Processing HDF5 file with index {i} ({i+1}/{self.num_parts})")
            
            # Start timing for this part
            if self.logger:
                self.logger.start_timer(f"process_part_{i}")
            
            # Prepare raw data for this part
            self.prepare_raw_data(hdf5_index=i)
            
            # Process signals
            self.process_signals(hdf5_index=i)
            
            # Compose dataset
            self.compose_dataset()
            
            # Clean intermediate files
            self.clean_intermediate_files(hdf5_index=i)
            
            # Stop timing for this part
            if self.logger:
                self.logger.stop_timer(f"process_part_{i}")
        
        # Clean up temp directory
        self.clean_temp_dir()
        
        # Log completion
        self.log('info', "Preprocessing completed successfully")
        self.log('info', f"ECG dataset saved to: {self.ecg_dataset_path}")
        self.log('info', f"Metadata saved to: {self.meta_dataset_path}")
        
        # Save preprocessing info
        self.save_preprocessing_info()

    
    def prepare_raw_data(self, hdf5_index: Optional[int] = None) -> None:
        """
        Prepare raw data by filtering the exams.csv file for the specified HDF5 part.
        
        Args:
            hdf5_index: Index of the HDF5 part to process
        """
        if hdf5_index is None:
            return
        
        self.log('info', f"Preparing raw data for HDF5 part {hdf5_index}")
        
        # Filter records.csv to select only records related to the current HDF5 part
        input_file = os.path.join(self.raw_data_path, 'exams.csv')
        if not os.path.exists(input_file):
            error_msg = f"File exams.csv not found in {self.raw_data_path}"
            self.log('error', error_msg)
            raise FileNotFoundError(error_msg)
            
        df = pd.read_csv(input_file)
        filtered_df = df[df['trace_file'] == f'exams_part{hdf5_index}.hdf5']
        output_file = os.path.join(self.temp_dir, 'records.csv')
        filtered_df.to_csv(output_file, index=False)
        
        self.log('info', f"records.csv file created with {len(filtered_df)} rows")

    
    def process_signals(self, hdf5_index: Optional[int] = None) -> None:
        """
        Process ECG signals using external scripts.
        
        Args:
            hdf5_index: Index of the HDF5 part to process
        """
        if hdf5_index is None:
            hdf5_index = 0
            
        hdf5_index_str = str(hdf5_index)
        self.log('info', f"Processing signals for HDF5 part {hdf5_index_str}")
        
        manifest_file = os.path.join(self.temp_dir, 'manifest.csv')
        
        # 1. Signal preprocessing
        self.log('debug', "Starting signal preprocessing script")
        self._execute_command([
            'python', self.signals_script,
            '--processed_root', self.temp_dir,
            '--raw_root', self.raw_data_path,
            '--manifest_file', manifest_file,
            '--hdf5_index', hdf5_index_str,
        ])
        
        # 2. Dataset splitting
        self.log('debug', "Starting dataset splitting script")
        self._execute_command([
            "python", self.splits_script,
            "--strategy", "random",
            '--processed_root', self.temp_dir,
        ])
        
        # 3. Label extraction
        self.log('debug', "Starting label extraction script")
        os.makedirs(os.path.join(self.temp_dir, "labels"), exist_ok=True)
        self._execute_command([
            "python", self.labels_script,
            "--processed_root", self.temp_dir,
        ])
        
        self.log('info', f"Signal processing completed for HDF5 part {hdf5_index_str}")
    

    def process_metadata(self) -> None:
        """
        For Code-15, this method is not necessary as metadata 
        is processed during dataset composition.
        """
        self.log('debug', "Skipping separate metadata processing for Code-15 dataset")
        pass

    
    def compose_dataset(self) -> None:
        """
        Compose two datasets:
        - meta_dataset: includes exam_id, age, sex, all classification labels
        - ecg_org_dataset: includes exam_id, original ECG [12, 2500]
        
        Handles all available labels, not just a single one.
        """
        self.log('info', 'Composing dataset...')
        
        # Track operation time
        if self.logger:
            self.logger.start_timer("compose_dataset")
        
        # 1. Load segmented ECG data
        segmented_split_file = os.path.join(self.temp_dir, 'segmented_split.csv')
        df = pd.read_csv(segmented_split_file)
        
        self.log('debug', f"Loaded segmented split file with {len(df)} records")
        
        dataset = {}
        for _, row in df.iterrows():
            path_to_data = row['path']
            dataset[path_to_data] = {
                'idx': int(row['idx']),
                'ecg': self._get_ecg_tensor(path_to_data),
            }
        
        self.log('debug', f"Loaded {len(dataset)} ECG tensors from files")
        
        # 2. Load metadata
        meta_file = os.path.join(self.temp_dir, 'meta.csv')
        df = pd.read_csv(meta_file)
        
        self.log('debug', f"Loaded metadata file with {len(df)} records")
        
        # Discover all available labels if not provided
        if self.labels is None:
            # Find all binary label columns (columns with only 0 and 1 values)
            potential_label_columns = []
            for col in df.columns:
                if set(df[col].unique()).issubset({0, 1, True, False}):
                    if col not in ['idx', 'exam_id', 'is_male']:  # Skip obvious non-label columns
                        potential_label_columns.append(col)
            
            self.labels = potential_label_columns
            self.log('info', f"Discovered {len(self.labels)} labels: {self.labels}")
        
        meta = {}
        for _, row in df.iterrows():
            meta_entry = {
                'exam_id': row['exam_id'],
                'is_male': 1 if row['is_male'] else 0,
                'age': row['age'],
            }
            
            # Add all label values
            for label in self.labels:
                if label in row:
                    meta_entry[label] = 1 if row[label] else 0
            
            meta[row['idx']] = meta_entry
        
        self.log('debug', f"Processed metadata for {len(meta)} records")
        
        # 3. Combine ECG and metadata
        records_with_metadata = 0
        for path in dataset.keys():
            idx = dataset[path]['idx']
            if idx in meta:
                # Add all metadata fields
                for key, value in meta[idx].items():
                    dataset[path][key] = value
                records_with_metadata += 1
            else:
                self.log('warning', f"Warning: {idx} not found in metadata.")
        
        self.log('debug', f"Combined data: {records_with_metadata} records with metadata")
        
        # 4. Convert to DataFrame
        # Include only ECG and exam_id for ECG dataset
        ecg_columns = ['idx']
        for path in dataset:
            if 'exam_id' in dataset[path]:
                ecg_columns.append('exam_id')
                break
        
        ecgs_df = pd.DataFrame(list(dataset.values()))
        meta_df = pd.DataFrame(list(dataset.values()))
        
        # Drop appropriate columns for each dataset
        meta_columns = [col for col in meta_df.columns if col != 'ecg']
        ecgs_df = ecgs_df[['exam_id', 'ecg']]
        meta_df = meta_df[meta_columns]
        
        self.log('debug', f"Created DataFrames: ECG ({len(ecgs_df)} rows), Metadata ({len(meta_df)} rows)")
        
        # 5. Handle ECG dataset (Pickle)
        if os.path.exists(self.ecg_dataset_path):
            self.log('debug', f"Appending to existing ECG dataset at {self.ecg_dataset_path}")
            existing_ecgs_df = pd.read_pickle(self.ecg_dataset_path)
            combined_ecgs_df = pd.concat([existing_ecgs_df, ecgs_df], ignore_index=True)
        else:
            self.log('debug', f"Creating new ECG dataset at {self.ecg_dataset_path}")
            combined_ecgs_df = ecgs_df
        combined_ecgs_df.to_pickle(self.ecg_dataset_path)
        
        # 6. Handle metadata (CSV)
        if os.path.exists(self.meta_dataset_path):
            self.log('debug', f"Appending to existing metadata at {self.meta_dataset_path}")
            existing_meta_df = pd.read_csv(self.meta_dataset_path)
            combined_meta_df = pd.concat([existing_meta_df, meta_df], ignore_index=True)
        else:
            self.log('debug', f"Creating new metadata file at {self.meta_dataset_path}")
            combined_meta_df = meta_df
        combined_meta_df.to_csv(self.meta_dataset_path, index=False)
        
        self.log('info', f"Dataset composed: {len(combined_ecgs_df)} ECG samples and metadata")
        
        label_columns = [col for col in combined_meta_df.columns if col not in ['idx', 'exam_id', 'is_male', 'age']]
        self.log('info', f"Available labels in metadata: {label_columns}")
        
        # Log class distribution
        if self.logger:
            class_distribution = {}
            for label in label_columns:
                if label in combined_meta_df.columns:
                    positive_count = combined_meta_df[label].sum()
                    total_count = len(combined_meta_df)
                    percentage = (positive_count / total_count) * 100
                    class_distribution[label] = f"{positive_count}/{total_count} ({percentage:.2f}%)"
            
            self.logger.log_result("Class Distribution", class_distribution)
            
        # Stop timing
        if self.logger:
            self.logger.stop_timer("compose_dataset")
    

    def clean_intermediate_files(self, hdf5_index: Optional[int] = None) -> None:
        """
        Remove intermediate files created during preprocessing.
        
        Args:
            hdf5_index: Index of the processed HDF5 part
        """
        self.log('info', f'Cleaning intermediate files for HDF5 with index {hdf5_index}...')
        
        # Remove intermediate directories for this step
        directories = ['labels', 'org', 'preprocessed', 'segmented']
        for directory in directories:
            dir_path = os.path.join(self.temp_dir, directory)
            if os.path.exists(dir_path):
                self.log('debug', f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path)
        
        # Remove intermediate files for this step
        files = ['records.csv', 'meta.csv', 'meta_split.csv', 'segmented.csv', 'segmented_split.csv']
        for file in files:
            file_path = os.path.join(self.temp_dir, file)
            if os.path.exists(file_path):
                self.log('debug', f"Removing file: {file_path}")
                os.remove(file_path)
        
        # Remove manifest.csv
        manifest_path = os.path.join(self.temp_dir, 'manifest.csv')
        if os.path.exists(manifest_path):
            self.log('debug', f"Removing manifest file: {manifest_path}")
            os.remove(manifest_path)
        
        self.log('info', f"Intermediate files cleanup completed")
    

    def _get_ecg_tensor(self, file_path: str) -> torch.Tensor:
        """
        Load a MAT file and convert ECG data to a PyTorch tensor.
        
        Args:
            file_path: Path to the MAT file containing ECG data
        
        Returns:
            PyTorch tensor containing ECG data
        """
        try:
            input_mat_data = scipy.io.loadmat(file_path)
            ecg_key = "feats"  # Key for the ECG waveform
            input_ecg_signal = input_mat_data[ecg_key]  # Extract the ECG waveform
            ecg_tensor = torch.tensor(input_ecg_signal, dtype=torch.float32)
            return ecg_tensor
        except Exception as e:
            self.log('error', f"Error loading ECG file {file_path}: {str(e)}")
            raise
    

    def _execute_command(self, cmd: List[str]) -> Tuple[str, str]:
        """
        Execute a system command and return the output.
        
        Args:
            cmd: List containing the command and its arguments
        
        Returns:
            Tuple containing stdout and stderr of the executed command
        """
        cmd_str = " ".join(cmd)
        self.log('debug', f"Executing: {cmd_str}")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()
        
        # Log command output
        if stdout_str:
            for line in stdout_str.split('\n'):
                if line.strip():
                    self.log('debug', f"STDOUT: {line}")
                    
        if stderr_str:
            for line in stderr_str.split('\n'):
                if line.strip():
                    self.log('warning', f"STDERR: {line}")
        
        # Check if command was successful
        if process.returncode != 0:
            self.log('error', f"Command failed with return code {process.returncode}: {cmd_str}")
        
        return stdout_str, stderr_str
    