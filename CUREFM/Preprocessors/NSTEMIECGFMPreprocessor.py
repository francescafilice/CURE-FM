import os
import pandas as pd
import torch
from typing import List, Optional
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d

from .BasePreprocessor import BasePreprocessor



class NSTEMIECGFMPreprocessor(BasePreprocessor):
    """
    Specific preprocessor for the NSTEMI ECG dataset with ECG-FM model.
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
        
        # Specific parameters for NSTEMI
        self.labels = labels
        self.primary_label = primary_label
        
        # Get percentage of data to process 
        self.dataset_percentage = dataset_percentage
        if self.dataset_percentage <= 0 or self.dataset_percentage > 1:
            self.log('warn', f"Invalid percentage value {self.dataset_percentage}, using 1.0 instead")
            self.dataset_percentage = 1.0
    
        # Additional logging for NSTEMI specific parameters
        self.log('debug', f"Labels to extract: {self.labels}")
        self.log('debug', f"Primary label: {self.primary_label}")
        self.log('debug', f"Processing {self.dataset_percentage * 100}% of the dataset")
        self.log('debug', f"percentage of patients whose ECGs must be preprocessed: {self.dataset_percentage}")


    def preprocess(self) -> None:
        """
        Execute preprocessing of the NSTEMI dataset.
        """
        self.log('info', "Starting preprocessing of the NSTEMI dataset")
        
        # Remove existing datasets if present
        if os.path.exists(self.ecg_dataset_path):
            self.log('info', f"Removing existing ECG dataset: {self.ecg_dataset_path}")
            os.remove(self.ecg_dataset_path)
        if os.path.exists(self.meta_dataset_path):
            self.log('info', f"Removing existing metadata dataset: {self.meta_dataset_path}")
            os.remove(self.meta_dataset_path)

        # todo: consentire di processare solo una parte del dataset

        # iter all ECG xml and preprocess only those appearing in NSTEMI csv
        self.compose_dataset()

        # Log completion
        self.log('info', "Preprocessing completed successfully")
        
        # Save preprocessing info
        self.save_preprocessing_info()
        
    
    def process_signals(self, exam_id: str) -> torch.Tensor:
        """
        Process ECG signal data for a given exam ID.
        
        Args:
            exam_id: The exam ID to process
            
        Returns:
            torch.Tensor: Processed ECG signal with shape [12, 2500]
        """
        
        self.log('debug', f"Processing ECG signal for exam ID: {exam_id}")
        
        # Construct the path to the MATLAB file
        signal_path = os.path.join(self.raw_data_path, 'signals', f"{exam_id}.mat")
        
        if not os.path.exists(signal_path):
            self.log('error', f"Signal file not found: {signal_path}")
            raise FileNotFoundError(f"Signal file not found for exam ID {exam_id}")
        
        try:
            mat_data = sio.loadmat(signal_path)
            
            sampling_rate = float(mat_data.get('sampling_rate', 500.0))
            self.log('debug', f"Detected sampling rate: {sampling_rate} Hz")
            
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            ecg_signal = np.zeros((12, 0))
        
            leads_data = []
            lead_shape = None
            
            for i, lead in enumerate(lead_names):
                if lead in mat_data:
                    lead_data = mat_data[lead]
                    
                    if lead_data.shape[0] == 1:
                        lead_data = lead_data.squeeze(0)
                        
                    # Store for consistency check
                    if lead_shape is None:
                        lead_shape = lead_data.shape
                    elif lead_shape != lead_data.shape:
                        self.log('warn', f"Lead {lead} has shape {lead_data.shape}, expected {lead_shape}")
                    
                    leads_data.append(lead_data)
                else:
                    self.log('error', f"Lead {lead} not found in signal file for exam {exam_id}")
                    raise ValueError(f"Missing lead {lead} in signal file")
            
            # Stack all leads into a single array [12, samples]
            ecg_signal = np.stack(leads_data)
            if ecg_signal.shape[0] != 12:
                self.log('error', f"Expected 12 leads, but found {ecg_signal.shape[0]} leads")
                raise ValueError(f"Unexpected number of leads: {ecg_signal.shape[0]}, expected 12")

            # ---------------------------------
            # preprocess ecg via ECG-FM specific preprocessing
            processed_data = self.preprocess_mat(
                {"path": signal_path},  
                desired_sample_rate=500.0,  
                standardize=True, 
                constant_lead_strategy='zero'  
            )
            ecg_signal = processed_data["feats"]
            # ---------------------------------
            
            # Trim the signal to the first 5 seconds
            seconds = 5
            samples_for_5_seconds = int(seconds * sampling_rate)
            if ecg_signal.shape[1] >= samples_for_5_seconds:
                ecg_signal = ecg_signal[:, :samples_for_5_seconds]

            ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32)
            
            self.log('debug', f"Processed ECG tensor shape: {ecg_tensor.shape}")
            return ecg_tensor
        
        except Exception as e:
            self.log('error', f"Error processing signal file {signal_path}: {str(e)}")
            raise
    

    def process_metadata(self) -> None:
        """
        Process and combine metadata of patients relative to each ECG exam.
        Only processes a subset based on the percentage parameter.
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
            metadata_df = merged_df[self.labels]

            # Preserve numeric types by ensuring they remain as their original types
            for col in metadata_df.columns:
                if metadata_df[col].dtype == 'int64' or metadata_df[col].dtype == 'int32':
                    metadata_df[col] = metadata_df[col].astype('int')
                elif metadata_df[col].dtype == 'float64' or metadata_df[col].dtype == 'float32':
                    metadata_df[col] = metadata_df[col].astype('float')
            
            # Check if file exists and remove if it does
            if os.path.exists(self.meta_dataset_path):
                self.log('info', f"Removing existing metadata file: {self.meta_dataset_path}")
                os.remove(self.meta_dataset_path)
            
            # Save the metadata to the specified path with type preservation
            metadata_df.to_csv(self.meta_dataset_path, index=False, float_format='%.8g')
            self.log('info', f"Saved merged metadata to {self.meta_dataset_path} with {len(metadata_df)} records")
            
        except Exception as e:
            self.log('error', f"Error during metadata processing: {str(e)}")
            raise
        
        # Return the exam IDs to be processed
        return metadata_df['exam_id'].tolist()
    

    def compose_dataset(self) -> None:
        """
        Compose two datasets:
        - meta_dataset: includes exam_id, sex, etc...
        - ecg_org_dataset: includes exam_id, original ECG [12, 2500]
        
        Handles all available labels, not just a single one.
        Processes only a portion of the dataset based on the percentage parameter.
        """
         
        self.log('info', f'Composing dataset (processing {self.dataset_percentage*100:.1f}% of patients)...')

        # Track operation time
        if self.logger:
            self.logger.start_timer("compose_dataset")
        
        # 1. compose meta dataset and get exam IDs to process (according to dataset_percentage)
        exam_ids_to_process = self.process_metadata()

        # 2. preprocess ECG signals for the selected exam IDs
        self.log('info', f"Processing ECG signals for {len(exam_ids_to_process)} exams...")
        try:
            # Create a dictionary to store processed ECG signals
            ecg_tensors = {}
            total_exams = len(exam_ids_to_process)
            
            # Process each exam ID
            for idx, exam_id in enumerate(tqdm(exam_ids_to_process, desc="Processing ECGs")):
                try:
                    # Process this exam's ECG signal
                    ecg_tensor = self.process_signals(exam_id)
                    
                    # Store the tensor in our dictionary
                    ecg_tensors[exam_id] = ecg_tensor
                    
                except Exception as e:
                    self.log('warn', f"Failed to process ECG for exam ID {exam_id}: {str(e)}")
                    continue


            # Create a DataFrame from the collected tensors
            signals_df = pd.DataFrame({
                'exam_id': list(ecg_tensors.keys()),
                'ecg': list(ecg_tensors.values())
            })
            
            # Save the DataFrame as pickle file
            self._save_signals_to_pkl(signals_df)
            
            self.log('info', f"Successfully processed {len(signals_df)} ECG signals")
            
        except Exception as e:
            self.log('error', f"Error processing ECG signals: {str(e)}")
            raise

        # Stop timing
        if self.logger:
            self.logger.stop_timer("compose_dataset")

        self.log('info', 'Dataset composition completed')
    

    def _save_signals_to_pkl(self, signals_df: pd.DataFrame):
        """
        Save processed ECG signals DataFrame to a .pkl file.
        Deletes existing file if present.
        
        Args:
            signals_df: DataFrame of processed ECG signals.
        """
        if os.path.exists(self.ecg_dataset_path):
            self.log('info', f"Removing existing ECG dataset file: {self.ecg_dataset_path}")
            os.remove(self.ecg_dataset_path)
        
        signals_df.to_pickle(self.ecg_dataset_path)
        self.log('info', f"Saved {len(signals_df)} processed ECG signals to {self.ecg_dataset_path}")


    def prepare_raw_data(self) -> None:
        # not needed for NSTEMI dataset
        pass


    def clean_intermediate_files(self) -> None:
        # not needed for NSTEMI dataset
        pass

    

    # ----------------------------------------------------------
    # ECG-FM specific preprocessing methods

    def resample(self, feats, curr_sample_rate, desired_sample_rate):
        """
        Resample an ECG using linear interpolation.
        """
        if curr_sample_rate == desired_sample_rate:
            return feats

        desired_sample_size = int(
            feats.shape[-1] * (desired_sample_rate / curr_sample_rate)
        )

        x = np.linspace(0, desired_sample_size - 1, feats.shape[-1])

        return interp1d(x, feats, kind='linear')(np.arange(desired_sample_size))


    def lead_std_divide(self, feats, constant_lead_strategy='zero'):
        # Calculate standard deviation along axis 1, keep dimensions for broadcasting
        std = feats.std(axis=1, keepdims=True)
        std_zero = std == 0

        # Check if there are any zero stds or if strategy is 'nan'
        if not std_zero.any() or constant_lead_strategy == 'nan':
            # Directly divide, which will turn constant leads into NaN if any
            feats = feats / std

            return feats, std

        # Replace zero standard deviations with 1 temporarily to avoid division by zero
        std_replaced = np.where(std_zero, 1, std)
        feats = feats / std_replaced

        if constant_lead_strategy == 'zero':
            # Replace constant leads to be 0
            zero_mask = np.broadcast_to(std_zero, feats.shape)
            feats[zero_mask] = 0

        elif constant_lead_strategy == 'constant':
            # Leave constant leads as is
            pass

        else:
            raise ValueError("Unexpected constant lead strategy.")

        return feats, std


    def preprocess_mat(self, row, desired_sample_rate, standardize, constant_lead_strategy):
        data = loadmat(row["path"])
        
        # Define the standard 12 lead names in order
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Extract leads data
        leads_data = []
        for lead in lead_names:
            if lead in data:
                lead_data = data[lead]
                # Check shape and squeeze if necessary (removing dimensions of size 1)
                if lead_data.shape[0] == 1:
                    lead_data = lead_data.squeeze(0)
                leads_data.append(lead_data)
            else:
                self.log('error', f"Lead {lead} not found in signal file")
                raise ValueError(f"Missing lead {lead} in signal file")
        
        # Stack all leads into a single array [12, samples]
        feats = np.stack(leads_data)
        
        # Get sampling rate from data or use default
        curr_sample_rate = float(data.get('sampling_rate'))
        
        # Clean up unnecessary fields
        if "__header__" in data: del data["__header__"]
        if "__version__" in data: del data["__version__"]
        if "__globals__" in data: del data["__globals__"]

        # 1. Resample
        feats = self.resample(feats, curr_sample_rate, desired_sample_rate)

        # 2. Standardize
        mean = None
        std = None
        if standardize:
            mean = feats.mean(axis=1, keepdims=True)
            feats = feats - mean
            feats, std = self.lead_std_divide(
                feats,
                constant_lead_strategy=constant_lead_strategy,
            )

        # Update data dictionary with processed values
        processed_data = {
            "feats": feats,
            "curr_sample_rate": desired_sample_rate, 
            "curr_sample_size": feats.shape[-1],
            "mean": mean,
            "std": std
        }
        
        return processed_data
