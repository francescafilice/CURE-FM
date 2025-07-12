import sys
import argparse
from CUREFM.utils.ConfigReader import ConfigReader
from CUREFM.utils.logger import Logger
from CUREFM.Preprocessors.PreprocessorFactory import PreprocessorFactory
from CUREFM.EmbeddingExtractors.EmbedderFactory import EmbedderFactory
from CUREFM.Classify.DataSampler import DataSampler
from CUREFM.Classify.ClassifiersHandler import ClassifiersHandler


def parse_arguments():
    parser = argparse.ArgumentParser(description='ECG-FM Processing Pipeline')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML configuration file')
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    
    # Initialize the configuration reader
    config = ConfigReader(args.config) 

    # Validate configuration
    if not config.validate():
        print("Invalid configuration. Terminating.")
        sys.exit(1)
    
    # Initialize the logger
    log_dir = config.classification_output_dir
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
        
    # Embedding extraction phase
    if config.extract_embeddings:
        logger.log_step("Embedding Extraction", 
                       f"Dataset: {config.dataset_type}, Model: {config.model_type}")
        logger.start_timer("embedding_extraction")
            
        # Create embedder
        embedder = EmbedderFactory.create_embedder(model_type=config.model_type,
                                                    dataset_type=config.dataset_type,
                                                    id_column = config.data_sampler_params.get('id_column'),
                                                    processed_data_dir=config.processed_data_dir,
                                                    output_dir=config.embeddings_dir,
                                                    checkpoint_path=config.ckpt_path,
                                                    ecg_dataset_path=config.ecg_dataset_path,
                                                    meta_dataset_path=config.meta_dataset_path,
                                                    logger=logger,  
                                                    **config.embedding_params)

        embedder.process_all(pool_methods=config.pool_methods)
        logger.stop_timer("embedding_extraction")
        logger.info("Embedding extraction completed successfully")
        
    # Classification phase
    if config.run_classification:
        logger.log_step("Classification", "Training and evaluating classifiers")
        logger.start_timer("classification")
        
        # Initialize data sampler
        sampler = DataSampler(config.data_sampler_params, logger=logger)
        
        # Initialize classifier
        classifier = ClassifiersHandler(sampler, 
                                        config.classification_params, 
                                        results_dir=config.classification_output_dir,
                                        logger=logger)
        
        # Train all classifiers
        logger.info("Starting classifier training")
        all_results = classifier.train_all()
        
        # Log summary of results
        logger.log_result("Classification Results Summary", {
            "Total target labels": len(config.data_sampler_params.get('target_labels', [])),
            "Total pool methods": len(config.pool_methods),
            "Total classifiers": len(config.classification_params.get('classifiers', []))
        })
        
        logger.stop_timer("classification")
        logger.info("Classifier training completed")
    
    # Log memory usage at the end
    logger.log_memory_usage()
    logger.info("All operations completed successfully!")


if __name__ == "__main__":
    main()
    # to run the script, use the command:
    # python main_embedding.py --config percorso/al/tuo/file_configurazione.yaml