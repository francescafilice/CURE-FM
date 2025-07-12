import os
import torch
import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
import scipy.signal as signal
from typing import List, Optional
from tqdm import tqdm

from .BasePreprocessor import BasePreprocessor


class NSTEMIHuBERTECGPreprocessor(BasePreprocessor):
    """
    Specific preprocessor for the NSTEMI ECG dataset with HuBERT-ECG model.
    Implements the methods of the BasePreprocessor base class.
    """
    
    def __init__(self,
                 raw_data_path: str,
                 processed_data_dir: str,
                 ecg_dataset_path: str,
                 meta_dataset_path: str,
                 labels: Optional[List[str]] = None,
                 primary_label: str = None,
                 dataset_percentage: float = 1.0,
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
            dataset_percentage: Percentage of dataset to process (1.0 = full dataset)
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
            dataset_percentage=dataset_percentage,
            **kwargs
        )
        
        # Specific parameters for NSTEMI with HuBERT-ECG
        self.labels = labels
        self.primary_label = primary_label
        
        # Get percentage of data to process 
        self.dataset_percentage = dataset_percentage
        if self.dataset_percentage <= 0 or self.dataset_percentage > 1:
            self.log('warn', f"Invalid percentage value {self.dataset_percentage}, using 1.0 instead")
            self.dataset_percentage = 1.0
            
        os.makedirs(processed_data_dir, exist_ok=True)
        
        self.log('debug', f"Labels to extract: {self.labels}")
        self.log('debug', f"Primary label: {self.primary_label}")
        self.log('debug', f"Processing {self.dataset_percentage * 100}% of the dataset")
    
    
    def filter(self, ecg: np.ndarray, original_fs: int) -> np.ndarray:
        """
        Apply bandpass filter to ECG signal (0.05-47 Hz).
        
        Args:
            ecg: ECG signal of shape [12, samples]
            
        Returns:
            Filtered ECG signal
        """
        lowcut = 0.05
        highcut = 47.0
        filter_order = int(0.3 * original_fs)
        numtaps = filter_order + 1

        # Normalize cutoff frequencies to Nyquist frequency
        nyq = 0.5 * original_fs
        low = lowcut / nyq
        high = highcut / nyq

        fir_coeff = signal.firwin(numtaps, [low, high], pass_zero=False)
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

        original_n_samples = ecg.shape[1]      
        duration_seconds = original_n_samples / original_fs
        resampled_n_samples = int(duration_seconds * target_fs)

        resampled_ecg = np.array([signal.resample(lead, resampled_n_samples) for lead in ecg])
        
        return resampled_ecg
    

    def scaling(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Normalize ECG signal to range [-1, 1].
        
        Args:
            ecg_signal: ECG signal of shape [12, samples]
            
        Returns:
            Normalized ECG signal
        """
        min_vals = np.min(ecg_signal, axis=1, keepdims=True)
        max_vals = np.max(ecg_signal, axis=1, keepdims=True)
        smooth = 1e-8   # small value to avoid division by zero
        normalized = 2 * (ecg_signal - min_vals) / (max_vals - min_vals + smooth) - 1
        return normalized


    def slice(self, ecg: np.ndarray, original_fs: int) -> np.ndarray:
        """
        Extract a 5-second segment from ECG signal.
        
        Args:
            ecg: ECG signal of shape [12, samples]
            original_fs: Original sampling frequency in Hz
            
        Returns:
            Sliced ECG signal of shape [12, samples] (5 seconds at original sampling rate)
        """
        target_samples = int(5 * original_fs)
        return ecg[:, :target_samples]


    def process_signals(self, exam_ids: List[str]) -> pd.DataFrame:
        """
        Process ECG signals for a list of exam IDs using HuBERT-ECG preprocessing.
        
        Args:
            exam_ids: List of exam IDs to process
            
        Returns:
            DataFrame containing processed ECG data with columns 'exam_id' and 'ecg'
        """
        self.log('info', f"Processing ECG signals for {len(exam_ids)} exams...")
        
        # Initialize lists to store processed data
        exam_id_list = []
        ecg_list = []
        
        # Process each ECG signal
        for exam_id in tqdm(exam_ids, desc="Processing ECGs"):
            try:
                # Construct the path to the MATLAB file
                signal_path = os.path.join(self.raw_data_path, 'signals', f"{exam_id}.mat")
                
                if not os.path.exists(signal_path):
                    self.log('error', f"Signal file not found: {signal_path}")
                    continue
                
                mat_data = sio.loadmat(signal_path)
                sampling_rate = float(mat_data.get('sampling_rate'))
                lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                
                # Extract each lead from its dedicated field
                leads_data = []
                for lead in lead_names:
                    if lead in mat_data:
                        lead_data = mat_data[lead]
                        if lead_data.shape[0] == 1:
                            lead_data = lead_data.squeeze(0)
                        leads_data.append(lead_data)
                    else:
                        self.log('error', f"Lead {lead} not found in signal file for exam {exam_id}")
                        raise ValueError(f"Missing lead {lead} in signal file")
                
                # Stack all leads into a single array [12, samples]
                ecg = np.stack(leads_data)
                
                # slicing to get a 5-second segment
                ecg = self.slice(ecg, original_fs=sampling_rate)
                
                # filtering (0.05-47 Hz bandpass)
                ecg = self.filter(ecg, original_fs=sampling_rate)
                
                # resampling from original 500 Hz to 100Hz
                ecg = self.resampling(ecg, original_fs=sampling_rate)
                
                # scaling to [-1, 1]
                ecg = self.scaling(ecg)
                
                # Flatten the ECG signal to 1D (from [12, 500] to [6000])
                ecg = ecg.flatten()
                
                # Convert to PyTorch tensor as float32
                ecg_tensor = torch.tensor(ecg, dtype=torch.float32)
                
                # Store processed ECG and metadata
                exam_id_list.append(exam_id)
                ecg_list.append(ecg_tensor)
                
            except Exception as e:
                self.log('error', f"Error processing record {exam_id}: {str(e)}")
                continue
        
        # Create DataFrame from processed data
        processed_data = pd.DataFrame({
            'exam_id': exam_id_list,
            'ecg': ecg_list
        })
        
        self.log('info', f"Successfully processed {len(processed_data)} ECG signals")
        
        return processed_data


    def process_metadata(self) -> List[str]:
        """
        Process and combine metadata of patients relative to each ECG exam.
        Only processes a subset based on the percentage parameter.
        
        Returns:
            List of exam IDs to process
        """
        self.log('info', "Processing metadata files...")
        
        # Read exam mappings file
        patients_csv_path = os.path.join(self.raw_data_path, 'NSTEMI_anonymized.csv')
        exam_mappings_path = os.path.join(self.raw_data_path, 'exam_mappings.csv')
        self.log('debug', f"Reading exam mappings from {exam_mappings_path}")
        
        try:
            exam_mappings_df = pd.read_csv(exam_mappings_path)
            self.log('debug', f"Loaded exam mappings with {len(exam_mappings_df)} records")
            
            # Sample a percentage of records if percentage < 1.0
            if self.dataset_percentage < 1.0:
                original_size = len(exam_mappings_df)
                sample_size = int(original_size * self.dataset_percentage)
                exam_mappings_df = exam_mappings_df.sample(n=sample_size, random_state=42)
                self.log('info', f"Sampled {sample_size}/{original_size} records ({self.dataset_percentage*100:.1f}%)")
        except Exception as e:
            self.log('error', f"Failed to read exam mappings file: {str(e)}")
            raise

        # Read patient data file
        self.log('debug', f"Reading patient data from {patients_csv_path}")
        try:
            patients_df = pd.read_csv(patients_csv_path)
            self.log('debug', f"Loaded patient data with {len(patients_df)} records")
        except Exception as e:
            self.log('error', f"Failed to read patients CSV file: {str(e)}")
            raise
        
        # Merge the two dataframes on patient_id
        self.log('debug', "Merging exam mappings and patient data")
        try:
            merged_df = pd.merge(
                exam_mappings_df, 
                patients_df, 
                on='patient_id',
                how='inner'
            )
            
            # Select only the required columns
            available_columns = [col for col in self.labels if col in merged_df.columns]
            metadata_df = merged_df[available_columns]
            
            # Convert boolean columns to integers (True -> 1, False -> 0)
            bool_columns = metadata_df.select_dtypes(include=['bool']).columns
            for col in bool_columns:
                metadata_df[col] = metadata_df[col].astype(int)
            
            # Preserve numeric types by ensuring they remain as their original types
            for col in metadata_df.columns:
                if metadata_df[col].dtype == 'int64' or metadata_df[col].dtype == 'int32':
                    metadata_df[col] = metadata_df[col].astype('int')
                elif metadata_df[col].dtype == 'float64' or metadata_df[col].dtype == 'float32':
                    metadata_df[col] = metadata_df[col].astype('float')
            
            # Save the metadata to the specified path with type preservation
            metadata_df.to_csv(self.meta_dataset_path, index=False, float_format='%.8g')
            self.log('info', f"Saved merged metadata to {self.meta_dataset_path} with {len(metadata_df)} records")
            
            # Extract list of exam IDs to process
            exam_ids = merged_df['exam_id'].tolist()
            return exam_ids
            
        except Exception as e:
            self.log('error', f"Error during metadata processing: {str(e)}")
            raise


    def compose_dataset(self, processed_data=None, metadata=None) -> None:
        """
        Save processed ECG data to pickle file and metadata to CSV.
        
        Args:
            processed_data: DataFrame containing processed ECG signals with columns 'exam_id' and 'ecg'
            metadata: DataFrame containing metadata (unused parameter as metadata is already saved)
        """
        if processed_data is not None and not processed_data.empty:
            # Remove existing file if it exists instead of trying to load and append
            if os.path.exists(self.ecg_dataset_path):
                self.log('info', f"Removing existing ECG dataset file: {self.ecg_dataset_path}")
                os.remove(self.ecg_dataset_path)
                
            # Save to pickle file
            with open(self.ecg_dataset_path, 'wb') as f:
                pickle.dump(processed_data, f)
            self.log('info', f"Saved processed ECG data to {self.ecg_dataset_path} with shape {processed_data.shape}")


    def preprocess(self) -> None:
        """
        Execute complete preprocessing of the NSTEMI dataset for HuBERT-ECG model.
        """
        self.log('info', "Starting preprocessing pipeline")
        
        # Remove existing datasets if present
        if os.path.exists(self.ecg_dataset_path):
            self.log('info', f"Removing existing ECG dataset: {self.ecg_dataset_path}")
            os.remove(self.ecg_dataset_path)
        if os.path.exists(self.meta_dataset_path):
            self.log('info', f"Removing existing metadata dataset: {self.meta_dataset_path}")
            os.remove(self.meta_dataset_path)
            
        os.makedirs(os.path.dirname(self.ecg_dataset_path), exist_ok=True)
        
        # 1. Process metadata and get exam IDs to process
        exam_ids = self.process_metadata()
        self.log('info', f"Obtained {len(exam_ids)} exam IDs to process")
        
        # 2. Process ECG signals for the selected exam IDs
        processed_data = self.process_signals(exam_ids)
        
        # 3. Save processed data
        self.compose_dataset(processed_data)
        
        self.log('info', "Preprocessing completed")


    def prepare_raw_data(self) -> None:
        """
        Prepare raw data (not needed for NSTEMI dataset).
        """
        # Not needed for NSTEMI
        pass


    def clean_intermediate_files(self) -> None:
        """
        Remove intermediate files created during preprocessing (not needed for NSTEMI).
        """
        # Not needed for NSTEMI
        pass