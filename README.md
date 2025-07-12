# CURE-FM: Context-aware Use of data REpresentations with Foundation Models 🧠⚡

CURE-FM is a comprehensive framework for downstream tasks through foundation model embeddings and lightweight ML classifiers. The framework is designed to support various datasets and models with a modular and easily extensible architecture.
This repository contains the codebase for CURE-FM, currently applied for **electrocardiogram (ECG) classification**.

## Key Features 🔧

### 1. ECG Data Preprocessing 

The framework can be decomposed into three main phases: **dataset preprocessing**, **embedding extraction**, **classification**. CUREFM currently supports preprocessing over the [CODE-15](https://zenodo.org/records/4916206) dataset both for the [ECG-FM](https://arxiv.org/abs/2408.05178) and [HuBERT-ECG](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v1) foundation models. The process includes:

- Reading raw data
- Signal normalization and filtering
- Data segmentation
- Management of classification labels (support for multiple labels)
- Creation of structured datasets

### 2. Embedding Extraction 

After preprocessing, the framework can **extract embeddings using foundation models**:

- Support for models such as ECG-FM and HuBERT-ECG
- Saving embeddings as individual PyTorch tensors
- Generation of various aggregated representations (pooling)
- Support for multiple pooling methods (average, maximum, minimum, last)

### 3. Classification 

Once the embeddings have been extracted, they are given as input to the specified classifiers in order to perform both **binary** and **multiclass classification** tasks:

- the target labels are entirely customizable
- optimized training times given the compact size of the input embeddings

### 4. Flexible Configuration ⚙

The system uses a configuration-based approach that allows:

- Specification of all parameters in a single YAML file
- Simple reuse of configurations
- Granular control of processing phases
- Separate configuration for preprocessing and embedding extraction

## System Requirements 💻

General and mandatory requirements:
- Python 3.9
- PyTorch
- SciPy
- pandas
- NumPy
- PyYAML
- tqdm
- fairseq-signals (**only if you want to employ ECG-FM as embedding extractor**): please clone the [fairseq-signal github repository](https://github.com/Jwoo5/fairseq-signals) in the dedicated folder at `root/faiserq-signals/` and install as explained in their github page. You can also refer to [ECG-FM github page](https://github.com/bowang-lab/ECG-FM).


## How to Use the Framework 🚀

### 1. Creating the Configuration File 

The configuration file is essential in order to quickly modify the parameters you want to test your pipeline on. It is a YAML file which must present the following main components:

```yaml
# General configuration
dataset_type: "YourDatasetName"
model_type: "YourFoundationModel"  
raw_data_path: "/path/to/raw/data"                          # the directory which contains the raw data
processed_data_dir: "/path/to/processed/data/directory"     # the directory which contains the data after being preprocessed 

# Output paths
ecg_dataset_path: "/path/to/processed/data/directory/ecg_org_dataset.pkl"     # preprocessed ecg dataset saved as a .pkl file
meta_dataset_path: "/path/to/processed/data/directory/meta_dataset.csv"       # metadata for the ecg dataset saved as a .csv file
embeddings_dir: "/path/to/embeddings/directory"                               # path to the directory where the embeddings will be saved

# Phase control
preprocess_dataset: true
extract_embeddings: true
run_classification: true

# Preprocessing parameters
preprocessing_params:
  labels: ["normal_ecg", "1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"] 
  metadata: ["exam_id", "age", "is_male"]    # labels treated as metadata

# Embedding parameters
ckpt_path: "/path/to/checkpoint.pt"   # path to the foundation model checkpoint 
embedding_params:
  batch_size: 64
pool_methods: ["max", "lst"]          # pooling methods to pool the embeddings. Currently available: max pooling ("max"), last token pooling ("lst")

# Classification output directory
classification_output_dir: "/home/sbartucci/ECG-FM/output_prova"

# Data sampling parameters
data_sampler_params:
  embedding_paths:
    max: "/path/to/embeddings/directory/pooled/max_pooled.csv"      # path to max pooled embeddings csv file
    lst: "/path/to/embeddings/directory/pooled/lst_pooled.csv"      # path to last token pooled embeddings csv file
  label_path: "/path/to/processed/data/directory/meta_dataset.csv" 
  target_labels: ["label_1", "label_2"]   # target labels
  id_column: "..."                        # column that contains the ID of the ECG record
  sample_percentage: 0.5                  # Fraction of available data to be used (0, 1]
  balanced_sampling: true                 # True if balanced data sambling according to target label, false otherwise
  balanced_folds: true                    # True if each fold maintains the same target label distribution as the original dataset, false otherwise
  test_size: 0.2                          # Use 20% as test set (customizable between (0, 1])
  validation_size: 0.1                    # Use 10% as validation set (customizable between (0, 1])
  n_folds: 2                              # Number of folds for cross-validation
  random_seed: 42                         # Seed for reproducibility

# Classification parameters
classification_params:
  classifiers: 
    - type: "classifier_1"
      enabled: true         # set to false if you do not want to perform classification via this classifier
      extract_shap: true    # set to false if you do not want to extract most relevant featuers via SHAP for this classifier
      hyperparameters:      # variable depending on the classifier
        ...         

    - type: "classifier_2"
      ... 
```

**Note 1**: Different datasets, FMs and classifiers may require different parameters: hence, it may be necessary to add some ad-hoc parameters, depending on which datasets, FMs and classifiers you are currently using. If this is the case, you must update both the `config.yaml` file and the `ConfigReader` object accordingly.

**Note 2**: if you want to test either ECG-FM or HuBERT-ECG as embedding extractors, a series of template configuration files are provided in the `root/` folder.

### 2. FM checkpoints 

Select your FM chepckpoint file and put it under the `root/checkpoints/` folder.

### 3. Running the Framework ▶

Specify a custom configuration file, place it under `root/` and run the `main_embedding.py` like this:

```bash
python main_embedding.py --config path/to/configuration.yaml
```

### 4. Output 

After execution, the framework will generate:

- **Preprocessed datasets**: Pickle files containing ECG signals and CSV files with metadata
- **Embeddings**: Individual PyTorch tensors for each ECG
- **Aggregated representations**: CSV files containing features aggregated through various pooling methods

You can specify the directories where to save these output files in the `config.yaml` file.

## Add a new dataset/FM/classifier 📊

### What you have to do 

CUREFM was puposedly developed to be easily extensible and flexible. Depending on your own needs, here is what you have to do in order to employ the framework with a custom dataset, foundation model or classifier.

#### To support a new dataset: 
1. Create a new preprocessor class that extends `BasePreprocessor`
2. Implement the abstract methods assuring that the dataset is preprocessed according to the requirements of the foundation model you selected as embedding extractor
3. Update `ProcessorFactory` to support the new dataset

#### To support a new foundation model: 
1. Create a new extractor class that extends `BaseEmbedder`
2. Implement the abstract methods
3. Update `EmbedderFactory` to support the new model

#### To support a new classifier: 
1. Create a new extractor class that extends `BaseClassifier`
2. Implement the abstract methods
3. Update `ClassifierFactory` to support the new classifier

**Note**: do not forget to update the `config.yaml` accordingly!

## Advanced Features 🚀

- **Multi-label management**: Support for datasets with multiple labels rather than single ones
- **Incremental processing**: Saving and reusing intermediate assets
- **Unified temporary directories**: Centralized management of temporary files
- **Individual embedding storage**: Each ECG has its own embedding file identified by the exam ID
- **Configurable pooling**: Ability to specify which pooling methods to use

## Optional: fine-tuning for performance comparison ⚖️

The `main_finetune.py` file in the `root/` folder has been provided in order to traditionally fine-tune the selected FMs for comparison purposes.
As before, a config file may help to configure necessary hyperparameters.
To fine-tune one of the provided FM, run:

```bash
python main_finetune.py --config path/to/configuration_finetune.yaml
```
Remember to properly substitute the `configuration_finetune.yaml` file in accordance with the FM to finetune.
