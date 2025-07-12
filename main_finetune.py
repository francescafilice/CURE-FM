import random
import sys
import os
import argparse
import numpy as np
import torch
from finetuning.Preprocessors.PreprocessorFactory import PreprocessorFactory
from finetuning.Finetuners.FinetunerFactory import FinetunerFactory
from finetuning.utils.logger import Logger
from finetuning.utils.ConfigReader import ConfigReader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ECG-FM finetuning process')
    parser.add_argument('--config', type=str, 
                        help='Path to the configuration YAML file')
    return parser.parse_args()

def main():
    """
    Main function to run the finetuning process.
    """
    # Parse command line arguments
    args = parse_arguments()
    # Initialize the configuration reader
    config = ConfigReader(args.config)

    # Validate configuration
    if not config.validate():
        print("Invalid configuration. Terminating.")
        sys.exit(1)
        
    # Set seed for reproducibility
    set_seed(config.get('data_sampler_params').get('random_seed'))   
    
    # Initialize global logger
    log_dir = config.finetuning_output_dir
    logger = Logger(output_dir=log_dir, name=f"{config.dataset_type}_{config.model_type}")
    
    # Log the configuration
    logger.info("Starting ECG-FM processing pipeline")
    logger.log_config(config)
     
    # Preprocessing phase
    if config.preprocess_dataset:
        logger.log_step("Preprocessing", 
                       f"Dataset: {config.dataset_type}, Model: {config.model_type}")
        logger.start_timer("preprocessing")
            
        # Create preprocessor
        preprocessor = PreprocessorFactory.create_preprocessor(model_type=config.model_type,
                                                                dataset_type=config.dataset_type,
                                                                raw_data_path=config.raw_data_path,
                                                                processed_data_dir=config.processed_data_dir,
                                                                ecg_dataset_path=config.ecg_dataset_path,
                                                                meta_dataset_path=config.meta_dataset_path,
                                                                logger=logger,  
                                                                **config.preprocessing_params)
            
        # Execute preprocessing
        preprocessor.preprocess()
        logger.stop_timer("preprocessing")
        logger.info("Preprocessing completed successfully")
    
    # Finetune
    if config.finetune:
        # Initialize the base logger
        base_log_dir = config.get('finetuning_output_dir')
    
        # Get list of target labels to finetune on
        target_labels = config.get('data_sampler_params', {}).get('target_labels', [])
        if not target_labels:
            print("No target labels defined in configuration. Terminating.")
            sys.exit(1)
        
        print(f"Starting finetuning for {len(target_labels)} target labels: {target_labels}")
        
        # Process each target label
        all_metrics = {}
        
        for target_label in target_labels:
            print(f"\n{'='*80}\nProcessing target label: {target_label}\n{'='*80}")
            
            # Create label-specific output directory
            label_output_dir = os.path.join(base_log_dir, target_label)
            os.makedirs(label_output_dir, exist_ok=True)
            
            # Initialize label-specific logger
            logger = Logger(output_dir=label_output_dir, name=f"{config.dataset_type}_{config.model_type}_{target_label}")
            
            # Log the configuration
            logger.info(f"Starting ECG-FM finetuning pipeline for label: {target_label}")
            logger.log_config(config)
            
            # Create the finetuner with label-specific save directory
            try:
                # Update save directory for this specific label
                finetuning_params = config.get('finetuning_params', {}).copy()
                label_save_dir = os.path.join(finetuning_params.get('save_dir', "checkpoints"), target_label)
                finetuning_params['save_dir'] = label_save_dir
                
                # Create output directory for results
                label_results_dir = os.path.join(label_output_dir, 'results')
                os.makedirs(label_results_dir, exist_ok=True)
                
                finetuner = FinetunerFactory.create_finetuner(
                    model_type=config.model_type,
                    dataset_type=config.dataset_type,
                    processed_data_dir=config.raw_data_path,
                    checkpoint_path=config.ckpt_path,
                    save_dir=label_save_dir,
                    logger=logger,
                    ecg_dataset_path=config.ecg_dataset_path,
                    meta_dataset_path=config.meta_dataset_path,
                    data_sampler_params=config.get('data_sampler_params', {}),
                    finetuning_params=finetuning_params,
                    output_dir=label_results_dir
                )
                
                # Get finetuning parameters
                batch_size = finetuning_params.get('batch_size')
                epochs = finetuning_params.get('epochs')
                learning_rate = finetuning_params.get('learning_rate')
                
                # Run the complete finetuning pipeline for this target label
                logger.info(f"Starting finetuning for target label: {target_label}")
                metrics = finetuner.process_all(
                    target_label=target_label,
                    batch_size=batch_size,
                    epochs=epochs, 
                    learning_rate=learning_rate
                )
                
                # Store metrics for this label
                all_metrics[target_label] = metrics
                
                # Log final metrics
                logger.info(f"Finetuning for {target_label} completed successfully!")
                
            except Exception as e:
                logger.error(f"Error during finetuning for {target_label}: {str(e)}")
                print(f"Error processing {target_label}: {str(e)}")
                continue
        
        # Log overall completion
        print(f"\nFinetuning completed for all target labels: {list(all_metrics.keys())}")
        print(f"Output saved to: {base_log_dir}")
    

if __name__ == "__main__":
    main()
