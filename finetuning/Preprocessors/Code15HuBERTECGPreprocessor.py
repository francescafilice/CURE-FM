import os
import torch
import numpy as np
import yaml
import h5py
import scipy.signal as signal
import pandas as pd
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from tqdm import tqdm

from .BasePreprocessor import BasePreprocessor


class Code15HuBERTECGPreprocessor(BasePreprocessor):
    """
    Specific preprocessor for the Code-15 ECG dataset with HuBERT-ECG model.
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
                 metadata: Optional[List[str]] = None,
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
            metadata: List of additional metadata fields to extract
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
            metadata=metadata,
            **kwargs
        )
        
        # Specific parameters for Code-15 with HuBERT-ECG
        self.labels = labels
        self.primary_label = primary_label
        self.num_parts = num_parts
        self.metadata = metadata
        
        # Create processed data directory if it doesn't exist
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Additional logging for HuBERT-ECG specific parameters
        self.log('debug', f"Labels to extract: {self.labels}")
        self.log('debug', f"Metadata fields to extract: {self.metadata}")
        self.log('debug', f"Primary label: {self.primary_label}")
        self.log('debug', f"Number of HDF5 parts: {self.num_parts}")
        
    
    def filter(self, ecg: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to ECG signal (0.05-47 Hz).
        
        Args:
            ecg: ECG signal of shape [12, samples]
            
        Returns:
            Filtered ECG signal
        """
        # Sampling frequency
        fs = 400  # Hz

        # Bandpass filter parameters
        lowcut = 0.05
        highcut = 47.0
        filter_order = int(0.3 * fs)
        numtaps = filter_order + 1

        # Normalize cutoff frequencies to Nyquist frequency
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # Design the FIR bandpass filter
        fir_coeff = signal.firwin(numtaps, [low, high], pass_zero=False)

        # Apply the filter to each lead
        filtered_ecg = np.array([signal.filtfilt(fir_coeff, [1.0], lead) for lead in ecg])
        
        return filtered_ecg
    

    def resampling(self, ecg: np.ndarray, original_fs: int) -> np.ndarray:
        """
        Resample ECG signal from original frequency to 100 Hz.
        
        Args:
            ecg: ECG signal of shape [12, samples]
            original_fs: Original sampling frequency in Hz
            
        Returns:
            Resampled ECG signal
        """
        target_fs = 100  # Hz

        # Calculate number of samples after resampling
        original_n_samples = ecg.shape[1]
        duration_seconds = original_n_samples / original_fs
        resampled_n_samples = int(duration_seconds * target_fs)

        # Resample each lead
        resampled_ecg = np.array([signal.resample(lead, resampled_n_samples) for lead in ecg])
        
        return resampled_ecg
    

    def scaling(self, ecg_signal: np.ndarray, smooth=1e-8) -> np.ndarray:
        """
        Normalize ECG signal to range [-1, 1].
        
        Args:
            ecg_signal: ECG signal of shape [12, samples]
            smooth: Small value to avoid division by zero
            
        Returns:
            Normalized ECG signal
        """
        min_vals = np.min(ecg_signal, axis=1, keepdims=True)
        max_vals = np.max(ecg_signal, axis=1, keepdims=True)
        normalized = 2 * (ecg_signal - min_vals) / (max_vals - min_vals + smooth) - 1
        return normalized


    def slice(self, ecg: np.ndarray) -> np.ndarray:
        """
        Extract a 5-second segment from ECG signal.
        
        Args:
            ecg: ECG signal of shape [samples, 12]
            
        Returns:
            Sliced ECG signal of shape [2000, 12] (5 seconds at 400 Hz)
        """
        # Check where the ECG has real (nonzero) data and returns first and last index
        nonzero_indices = np.where(ecg > 0)[0]
        
        # Handle case where no positive values are found
        if len(nonzero_indices) == 0:
            # Try with absolute values instead
            nonzero_indices = np.where(np.abs(ecg) > 0)[0]
            
            # If still empty, just start from the beginning
            if len(nonzero_indices) == 0:
                start = 0
            else:
                start = nonzero_indices[0]
        else:
            start = nonzero_indices[0]
        
        # slice signal to 5 sec
        original_fs = 400  # original frequency
        duration = 5  # target seconds
        end_5_sec = start + original_fs * duration
        
        # Make sure we don't go beyond the signal length
        if end_5_sec > ecg.shape[0]:
            # If not enough data after start, use the end and go backwards
            end_5_sec = min(ecg.shape[0], start + original_fs * duration)
            start = max(0, end_5_sec - original_fs * duration)
        
        return ecg[start:end_5_sec, :]
    

    def process_signals(self, hdf5_index: Optional[int] = None):
        """
        Process ECG signals from the specified HDF5 file.
        
        Args:
            hdf5_index: Index of the HDF5 file to process
            
        Returns:
            Tuple containing processed ECG data and corresponding metadata
        """
            
        # Construct the HDF5 file path
        hdf5_file = f"{self.raw_data_path}/exams_part{hdf5_index}.hdf5"
        
        if not os.path.exists(hdf5_file):
            self.log('error', f"HDF5 file not found: {hdf5_file}")
            return None, None
        
        # Open HDF5 file
        with h5py.File(hdf5_file, 'r') as hdf:
            # Get the exam IDs and tracings
            # Note: last sample has non-existent exam_id 0 and has all-zero tracing, so we drop it
            exam_ids = hdf['exam_id'][:-1]
            tracings = hdf['tracings'][:-1]
                            
            # Initialize lists to store processed data
            exam_id_list = []
            ecg_list = []
            
            # Process each ECG signal
            for i, (exam_id, tracing) in tqdm(enumerate(zip(exam_ids, tracings)), total=len(exam_ids)):
                try:
                    # Use numpy array directly (ensuring float32 type)
                    ecg = tracing.astype(np.float32)
                    
                    # Apply slicing to get a 5-second segment
                    ecg = self.slice(ecg)
                    
                    # Transpose to shape [12, samples] for further processing
                    ecg = ecg.T
                    
                    # Apply filtering (0.05-47 Hz bandpass)
                    ecg = self.filter(ecg)
                    
                    # Apply resampling from 400Hz to 100Hz
                    ecg = self.resampling(ecg, original_fs=400)
                    
                    # Apply scaling to [-1, 1]
                    ecg = self.scaling(ecg)
                    
                    # Flatten the ECG signal to 1D (from [12, 500] to [6000])
                    ecg = ecg.flatten()
                    
                    # Convert to PyTorch tensor as float32 only at the end of processing
                    ecg_tensor = torch.tensor(ecg, dtype=torch.float32)
                    
                    # Store processed ECG and metadata
                    exam_id_list.append(exam_id)
                    ecg_list.append(ecg_tensor)
                    
                except Exception as e:
                    self.log('error', f"Error processing record {exam_id}: {str(e)}")
        
        # Create DataFrame from processed data
        processed_data = pd.DataFrame({
            'exam_id': exam_id_list,
            'ecg': ecg_list
        })
        
        self.log('info', f"Processed {len(processed_data)} records from {hdf5_file}")
        
        return processed_data
    

    def compose_dataset(self, processed_data, metadata) -> None:
        """
        Save processed ECG data to pickle file and metadata to CSV.
        
        Args:
            processed_data: DataFrame containing processed ECG signals with columns 'exam_id' and 'ecg'
            metadata: DataFrame containing metadata
        """
        # Save processed ECG data
        if processed_data is not None and not processed_data.empty:
            if os.path.exists(self.ecg_dataset_path):
                # If file exists, load existing data and append
                try:
                    with open(self.ecg_dataset_path, 'rb') as f:
                        existing_data = pickle.load(f)
                        processed_data = pd.concat([existing_data, processed_data], ignore_index=True)
                except Exception as e:
                    self.log('error', f"Error loading existing data: {str(e)}")
                    
            # Save to pickle file
            with open(self.ecg_dataset_path, 'wb') as f:
                pickle.dump(processed_data, f)
            self.log('info', f"Saved processed ECG data to {self.ecg_dataset_path} with shape {processed_data.shape}")
        
        # Save metadata
        if not metadata.empty:
            if os.path.exists(self.meta_dataset_path):
                # If file exists, load existing metadata and append
                existing_metadata = pd.read_csv(self.meta_dataset_path)
                metadata = pd.concat([existing_metadata, metadata], ignore_index=True)
            
            metadata.to_csv(self.meta_dataset_path, index=False)
            self.log('info', f"Saved metadata to {self.meta_dataset_path}")

    def preprocess(self) -> None:
        """
        Execute complete preprocessing of the Code-15 dataset for HuBERT-ECG model.
        """
        
        self.log('info', "Starting preprocessing pipeline")
        
        # Remove existing datasets if present
        if os.path.exists(self.ecg_dataset_path):
            self.log('info', f"Removing existing ECG dataset: {self.ecg_dataset_path}")
            os.remove(self.ecg_dataset_path)
        if os.path.exists(self.meta_dataset_path):
            self.log('info', f"Removing existing metadata dataset: {self.meta_dataset_path}")
            os.remove(self.meta_dataset_path)
            
        # Create necessary directories
        os.makedirs(os.path.dirname(self.ecg_dataset_path), exist_ok=True)
        
        # Clear existing dataset files to avoid appending to old data
        if os.path.exists(self.ecg_dataset_path):
            os.remove(self.ecg_dataset_path)
            self.log('info', f"Removed existing ECG dataset file: {self.ecg_dataset_path}")
        
        if os.path.exists(self.meta_dataset_path):
            os.remove(self.meta_dataset_path)
            self.log('info', f"Removed existing metadata file: {self.meta_dataset_path}")
        
        # Process each HDF5 part
        for i in range(17, 18):  # range(self.num_parts):  # TODO: ripristina per processare l'intero dataset
            self.log('info', f"Processing HDF5 file with index {i} ({i+1}/{self.num_parts})")
            processed_data = self.process_signals(hdf5_index=i)
            metadata = self.process_metadata(hdf5_index=i)
            self.compose_dataset(processed_data, metadata)
                    
        self.log('info', "Preprocessing completed")

    
    def process_metadata(self, hdf5_index: int) -> pd.DataFrame:
        hdf5_file = f"{self.raw_data_path}/exams_part{hdf5_index}.hdf5"
        with h5py.File(hdf5_file, 'r') as hdf:
            exam_ids = hdf['exam_id'][:]
        
        # List of columns to keep - use list concatenation or sets to combine lists
        columns_to_keep = self.labels + self.metadata

        exam_df = pd.read_csv(os.path.join(self.raw_data_path, 'exams.csv'))

        # Filter rows where 'exam_id' matches - use exam_ids directly
        filtered_df = exam_df[exam_df['exam_id'].isin(exam_ids)]
        self.log('info', f"Filtered metadata to {len(filtered_df)} rows matching HDF5 exam_ids")

        # Select only the columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_keep if col in filtered_df.columns]
        filtered_df = filtered_df[existing_columns]
        self.log('info', f"Selected columns: {existing_columns}, len: {len(filtered_df)}")
        
        # Convert boolean columns to integers (True -> 1, False -> 0)
        bool_columns = filtered_df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            filtered_df[col] = filtered_df[col].astype(int)

        return filtered_df
    

    def prepare_raw_data(self, hdf5_index: Optional[int] = None) -> None:
        """
        Prepare raw data by filtering the exams.csv file for the specified HDF5 part.
        
        Args:
            hdf5_index: Index of the HDF5 part to process
        """
        # Not needed for HuBERT-ECG
        pass

    
    def clean_intermediate_files(self, hdf5_index: Optional[int] = None) -> None:
        """
        Remove intermediate files created during preprocessing.
        
        Args:
            hdf5_index: Index of the processed HDF5 part
        """
        # Implement if temporary files are created during processing
        pass