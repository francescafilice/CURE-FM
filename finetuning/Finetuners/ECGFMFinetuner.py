from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List
import seaborn as sns
from math import ceil
from transformers import get_linear_schedule_with_warmup
import glob
from .BaseFinetuner import BaseFinetuner
from .DataSampler import DataSampler
from .tools.wave2vec2_cmsc_custom_ft_code15 import Wav2Vec2CMSCConfigCustomFtCode15, Wav2Vec2CMSCModelCustomFtCode15


class ECGDataset(torch.utils.data.Dataset):
    """
    Dataset for ECGs optimized for the ECG-FM model
    """
    def __init__(self, features, labels, random_crop=False, downsampling_factor=None):
        """
        Initialize the dataset with features and labels
        
        Args:
            features: DataFrame with ECG features
            labels: Series with binary labels
            random_crop: Whether to apply random crop as data augmentation
            downsampling_factor: Downsampling factor to apply to features
        """
        # Extract tensors
        tensor_list = []
        for i in range(len(features)):
            series = features.iloc[i]
            tensor_list.append(series[0])

        # Stack tensors into a single tensor of shape [N, 12, 2500]
        self.features = torch.stack(tensor_list)
        
        # Apply downsampling if necessary
        if downsampling_factor is not None:
            self.features = self.features[:, :, ::downsampling_factor]
            
        # Convert labels to tensors
        if isinstance(labels, pd.Series):
            self.labels = torch.tensor(labels.values, dtype=torch.float32).reshape(-1, 1)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        
        self.random_crop = random_crop
        
        print(f"Dataset initialized - Features: {self.features.shape}, Labels: {self.labels.shape}")
    
    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Returns features and labels for a given index
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple (features, label) for the specified index
        """
        # If requested, apply random crop
        if self.random_crop:
            # We assume the data is sampled at 500Hz
            crop_size = 2500
            if self.features.shape[2] > crop_size:
                start_idx = np.random.randint(0, self.features.shape[2] - crop_size)
                features = self.features[idx, :, start_idx:start_idx + crop_size]
            else:
                features = self.features[idx]
        else:
            features = self.features[idx]
            
        return features, self.labels[idx]


class ECGFMFinetuner(BaseFinetuner):
    """
    ECG-FM specific finetuner for ECG data.
    Implements the methods of the abstract BaseFinetuner class.
    """
    
    def __init__(self,
             processed_data_dir: str,
             ecg_dataset_path: str,
             meta_dataset_path: str,
             checkpoint_path: str,
             save_dir: str = "checkpoints",
             dataset_name: str = "code15",
             logger=None,
             output_dir=None,
             **kwargs):
        """
        Initialize the ECG-FM finetuner with the provided parameters
        
        Args:
            processed_data_dir: Directory containing processed data
            ecg_dataset_path: Path to the ECG dataset
            meta_dataset_path: Path to the metadata dataset
            checkpoint_path: Path to the pretrained model checkpoint
            save_dir: Directory to save fine-tuned model checkpoints
            dataset_name: Name of the dataset
            logger: Logger instance to track progress
            output_dir: Directory to save evaluation results
            **kwargs: Additional parameters
        """
        super().__init__(
            processed_data_dir=processed_data_dir,
            ecg_dataset_path=ecg_dataset_path,
            meta_dataset_path=meta_dataset_path,
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            logger=logger,
            **kwargs
        )
        
        # ECG-FM specific parameters
        self.dataset_name = dataset_name
        self.batch_size = kwargs.get('finetuning_params', {}).get('batch_size', 32)
        self.output_dir = output_dir
        
        # Specific parameters for ECG-FM, inspired by HuBERT
        self.transformer_blocks_to_unfreeze = kwargs.get('finetuning_params', {}).get('transformer_blocks_to_unfreeze', 0)
        self.unfreeze_conv_embedder = kwargs.get('finetuning_params', {}).get('unfreeze_conv_embedder', False)
        self.layer_wise_lr = kwargs.get('finetuning_params', {}).get('layer_wise_lr', False)
        self.weight_decay_mult = kwargs.get('finetuning_params', {}).get('weight_decay_mult', 1)
        self.model_dropout_mult = kwargs.get('finetuning_params', {}).get('model_dropout_mult', 0)
        self.downsampling_factor = kwargs.get('finetuning_params', {}).get('downsampling_factor', None)
        self.random_crop = kwargs.get('finetuning_params', {}).get('random_crop', False)
        self.dynamic_reg = kwargs.get('finetuning_params', {}).get('dynamic_reg', False)
        self.use_loss_weights = kwargs.get('finetuning_params', {}).get('use_loss_weights', False)
        
        # Constants for dynamic regularization
        self.EPS = 1e-9
        self.MINIMAL_IMPROVEMENT = 1e-3
        self.DROPOUT_DYNAMIC_REG_FACTOR = 0.05
        
        # Directory for fine-tuned checkpoints
        self.finetuned_dir = os.path.join(self.save_dir, f'{dataset_name}_ecgfm_finetuned')
        os.makedirs(self.finetuned_dir, exist_ok=True)
        
        # Log ECG-FM specific parameters
        self.log('debug', f"Dataset name: {self.dataset_name}")
        self.log('debug', f"Batch size: {self.batch_size}")
        self.log('debug', f"Transformer blocks to unfreeze: {self.transformer_blocks_to_unfreeze}")
        self.log('debug', f"Unfreeze conv embedder: {self.unfreeze_conv_embedder}")
        self.log('debug', f"Layer-wise LR: {self.layer_wise_lr}")
        self.log('debug', f"Random crop: {self.random_crop}")
        self.log('debug', f"Downsampling factor: {self.downsampling_factor}")
        self.log('debug', f"Finetuned directory: {self.finetuned_dir}")
        self.log('debug', f"Output directory: {self.output_dir}")
        
    def set_transformer_blocks_trainable(self, n_blocks: int) -> None:
        """
        Set the last n_blocks of the transformer as trainable.
        
        Args:
            n_blocks: Number of blocks to make trainable starting from the end
        """
        # Transformer layers are in encoder.layers directly in the main model
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            all_layers = self.model.encoder.layers
            total_layers = len(all_layers)
            
            self.log('info', f"The model has {total_layers} total transformer blocks")
            
            # Freeze all transformer blocks
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                
            # If n_blocks > 0, make the last n_blocks trainable
            if n_blocks > 0:
                start_idx = max(0, total_layers - n_blocks)
                for i in range(start_idx, total_layers):
                    for param in all_layers[i].parameters():
                        param.requires_grad = True
                        
                self.log('info', f"Unfrozen the last {n_blocks} transformer blocks (from {start_idx} to {total_layers-1})")
        else:
            self.log('warning', "Transformer structure not found, printing model structure")

    def set_feature_extractor_trainable(self, trainable: bool) -> None:
        """
        Set feature extraction layers as trainable or frozen.
        
        Args:
            trainable: True to make them trainable, False to freeze them
        """
        # The feature extractor is in feature_extractor directly in the model
        if hasattr(self.model, 'feature_extractor'):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = trainable
                
            self.log('info', f"Feature extractor set as {'trainable' if trainable else 'frozen'}")
        else:
            self.log('warning', "Feature extractor not found, printing model structure")
        
    def set_specific_layers_trainable(self, layer_names: List[str]) -> None:
        """
        Set specific model layers as trainable.
        
        Args:
            layer_names: List of layer names to make trainable
        """
        found_layers = 0
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                self.log('debug', f"Layer made trainable: {name}")
                found_layers += 1
        
        if found_layers > 0:
            self.log('info', f"Made {found_layers} specific layers trainable")
        else:
            self.log('warning', "None of the specified layers were found in the model")
    
    def dynamic_regularizer(self, optimizer, model, penalty):
        """
        Dynamically adjust the model's regularization
        
        Args:
            optimizer: The optimizer for the model
            model: The model to adjust
            penalty: Whether to apply a penalty or reduce regularization
        """
        if penalty:
            # Increase regularization
            optimizer.param_groups[0]['weight_decay'] *= 5
            for name, module in model.named_modules():
                if 'dropout' in name:
                    module.p += self.DROPOUT_DYNAMIC_REG_FACTOR
        else:
            # Reduce regularization
            optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'] / 5)
            for name, module in model.named_modules():
                if 'dropout' in name:
                    module.p = max(0.1, module.p - self.DROPOUT_DYNAMIC_REG_FACTOR)
    
    def build_model(self) -> None:
        """
        Build the ECG-FM model from checkpoint for fine-tuning.
        """
        self.log('info', f"Loading ECG-FM model from checkpoint: {self.checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create ECG-FM model
            model_config = Wav2Vec2CMSCConfigCustomFtCode15()
            self.model = Wav2Vec2CMSCModelCustomFtCode15(model_config)
            
            # Log model architecture
            self.log('debug', f"Model architecture: {type(self.model).__name__}")
        
            # Load weights from checkpoint
            self.model.load_state_dict(checkpoint['model'], strict=False)
            
            # Move model to correct device
            self.model.to(self.device)
            
            # Apply custom dropout operations
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Dropout):
                    original_p = module.p
                    module.p = original_p + self.DROPOUT_DYNAMIC_REG_FACTOR * self.model_dropout_mult
                    self.log('debug', f"Modified dropout in {name}: {original_p} → {module.p}")
            
            # Calculate model size
            model_size_mb = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            self.log('info', f"ECG-FM model loaded. Model size: {model_size_mb:.2f} MB")
            self.log('info', f"Total parameters: {total_params:,}")
            
            # By default initially freeze the entire model
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Configure feature extractor and transformer blocks
            self.set_feature_extractor_trainable(self.unfreeze_conv_embedder)
            self.set_transformer_blocks_trainable(self.transformer_blocks_to_unfreeze)
            
            # Make specific layers trainable
            specific_layers = [
                'post_extract_proj.weight',
                'post_extract_proj.bias',
                'conv_pos.pos_conv.0.bias',
                'conv_pos.pos_conv.0.parametrizations.weight.original0',
                'conv_pos.pos_conv.0.parametrizations.weight.original1',
                'layer_norm.weight',
                'layer_norm.bias'
            ]
            self.set_specific_layers_trainable(specific_layers)
            
            # Always make the classification layer trainable
            if hasattr(self.model, 'classification_head'):
                for param in self.model.classification_head.parameters():
                    param.requires_grad = True
                self.log('info', "Classification layer set as trainable")
            
            # Log memory usage
            if self.logger:
                self.logger.log_memory_usage()
                
        except Exception as e:
            error_msg = f"Unable to load ECG-FM model: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
    
    def prepare_data(self, target_label: str, batch_size: int = None) -> None:
        """
        Prepare data for fine-tuning by creating train/val/test splits.
        
        Args:
            target_label: Target label for binary classification
            batch_size: Batch size for training, if None use the one from config
        """
        self.target_label = target_label
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        # Initialize DataSampler with config from kwargs
        sampler_config = self.kwargs.get('data_sampler_params', {})
        self.log('info', f"Initializing DataSampler with target label: {target_label}")
        
        try:
            self.data_sampler = DataSampler(sampler_config)
            
            # Check if target label is available
            available_labels = self.data_sampler.get_all_target_labels()
            if target_label not in available_labels:
                error_msg = f"Target label '{target_label}' not found among available labels: {available_labels}"
                self.log('error', error_msg)
                raise ValueError(error_msg)
                
            # Create train/val/test splits
            self.log('info', f"Creating dataset splits for label: {target_label}")
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_sampler.create_train_val_test_split(target_label)
            
            self.log('info', f"Splits obtained - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples")
            
            # Calculate weights for loss balancing if needed
            if self.use_loss_weights:
                # Calculate weights for imbalanced classes
                pos_count = np.sum(y_train)
                neg_count = len(y_train) - pos_count
                pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                self.train_pos_weights = torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
                
                pos_count_val = np.sum(y_val)
                neg_count_val = len(y_val) - pos_count_val
                pos_weight_val = neg_count_val / pos_count_val if pos_count_val > 0 else 1.0
                self.val_pos_weights = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)
                
                self.log('info', f"Loss weights - Train: {pos_weight:.4f}, Val: {pos_weight_val:.4f}")
            else:
                self.train_pos_weights = None
                self.val_pos_weights = None
            
            # Create PyTorch datasets with parameters
            train_dataset = ECGDataset(
                X_train, y_train, random_crop=self.random_crop, downsampling_factor=self.downsampling_factor
            )
            val_dataset = ECGDataset(
                X_val, y_val, random_crop=False, downsampling_factor=self.downsampling_factor
            ) 
            test_dataset = ECGDataset(
                X_test, y_test, random_crop=False, downsampling_factor=self.downsampling_factor
            )
            
            num_workers = self.kwargs.get('finetuning_params', {}).get('num_workers', 4)
            pin_memory = self.kwargs.get('finetuning_params', {}).get('pin_memory', True)
            prefetch_factor = self.kwargs.get('finetuning_params', {}).get('prefetch_factor', 2)
            
            self.log('info', f"Using batch size: {batch_size}, workers: {num_workers}, "
                          f"pin_memory: {pin_memory}, prefetch_factor: {prefetch_factor}")
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor
            )
            
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            self.log('info', f"Data loaders created - Training: {len(self.train_loader)} batches, "
                          f"Validation: {len(self.val_loader)} batches, Test: {len(self.test_loader)} batches")
        
        except Exception as e:
            error_msg = f"Error in data preparation: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
    
    def finetune(self, epochs: int, learning_rate: float) -> None:
        """
        Fine-tune the ECG-FM model on the prepared data.
        
        Args:
            epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Verify that model and data are ready
        if self.model is None:
            error_msg = "Model not initialized. Call build_model() first."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
            
        if self.train_loader is None or self.val_loader is None:
            error_msg = "Data not prepared. Call prepare_data() first."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
        
        # Enable cudnn benchmark to optimize performance
        torch.backends.cudnn.benchmark = True
        
        # Get training parameters from kwargs
        weight_decay = 0.01 * self.weight_decay_mult
        patience = self.kwargs.get('finetuning_params', {}).get('patience', 5)
        accumulation_steps = self.kwargs.get('finetuning_params', {}).get('accumulation_steps', 1)
        betas = (0.9, 0.98)
        ramp_up_perc = self.kwargs.get('finetuning_params', {}).get('ramp_up_perc', 0.08)
        target_metric = self.kwargs.get('finetuning_params', {}).get('target_metric', 'auroc')
        
        self.log('info', f"Starting fine-tuning with {epochs} epochs, learning rate: {learning_rate}")
        
        # Set loss criterion
        if self.use_loss_weights:
            criterion_train = nn.BCEWithLogitsLoss(pos_weight=self.train_pos_weights)
            criterion_val = nn.BCEWithLogitsLoss(pos_weight=self.val_pos_weights)
        else:
            criterion_train = nn.BCEWithLogitsLoss()
            criterion_val = nn.BCEWithLogitsLoss()
        
        # Configure parameter groups to optimize learning rate for different layers
        parameters_group = self.configure_parameter_groups(learning_rate)
    
        
        # Create optimizer
        optimizer = optim.AdamW(
            parameters_group,
            betas=betas,
            eps=self.EPS,
            weight_decay=weight_decay,
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log('info', f"Total parameters: {total_params:,}")
        self.log('info', f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        # Calculate total training steps
        training_steps = epochs * len(self.train_loader) // accumulation_steps
        
        # Configure learning rate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(ramp_up_perc * training_steps),
            num_training_steps=training_steps,
        )
        
        # Configure scaler for mixed precision training
        scaler = torch.amp.GradScaler()
        
        # Initialize variables for early stopping
        patience_count = 0
        best_val_loss = float('inf')
        best_val_target_score = 0.0
        global_step = 0
        
        # Lists to store losses and metrics for plots
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Training/validation loop
        for epoch in range(epochs):
            self.model.train()
            
            self.log('info', f"Epoch {epoch+1}/{epochs}")
            
            batch_train_losses = []
            
            # Progress bar for training
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for data, labels in progress_bar:
                global_step += 1
                
                # Move tensors to device
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    outputs = self.model(source=data)
                    loss = criterion_train(outputs, labels)
                    
                    # Normalize loss for accumulation steps
                    loss = loss / accumulation_steps
                    
                # Backward pass with scaling for mixed precision
                scaler.scale(loss).backward()
                batch_train_losses.append(loss.item() * accumulation_steps)
                    
                # Update weights after accumulating gradients
                if global_step % accumulation_steps == 0:
                    # Clip gradient norm
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # Optimizer step with scaler for mixed precision
                    scaler.step(optimizer)
                    lr_scheduler.step()
                    scaler.update()
                    optimizer.zero_grad()
            
                progress_bar.set_postfix({
                    "train_loss": np.mean(batch_train_losses[-10:]) if batch_train_losses else 0.0
                })
                
                # Validation step
                if global_step % (len(self.train_loader) // 2) == 0:  # Validate twice per epoch
                    
                    self.model.eval()
                    
                    # Reset metrics
                    val_batch_losses = []
                    all_labels = []
                    all_preds = []
                    all_probs = []
                    
                    self.log('info', f"Validation at epoch {epoch+1}, step {global_step}")
                    
                    # Validation loop
                    for data, labels in tqdm(self.val_loader, desc="Validation"):
                        data = data.to(self.device)
                        labels = labels.to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(source=data)
                            loss = criterion_val(outputs, labels)
                        
                        val_batch_losses.append(loss.item())
                        
                        # Calculate binary predictions
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        preds = (probs >= 0.5).astype(int)
                        
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(preds)
                        all_probs.extend(probs)
                    
                    # Calculate validation statistics
                    val_loss = np.mean(val_batch_losses)
                    train_loss = np.mean(batch_train_losses)
                    
                    # Calculate metrics
                    all_labels = np.array(all_labels)
                    all_preds = np.array(all_preds)
                    all_probs = np.array(all_probs)
                    
                    accuracy = np.mean(all_preds == all_labels)
                    precision = precision_score(all_labels, all_preds, zero_division=0)
                    recall = recall_score(all_labels, all_preds, zero_division=0)
                    f1 = f1_score(all_labels, all_preds, zero_division=0)
                    try:
                        auroc = roc_auc_score(all_labels, all_probs)
                    except:
                        auroc = 0.5  # Fallback if there's only one class
                    
                    target_score_map = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auroc': auroc
                    }
                    
                    target_score = target_score_map.get(target_metric, auroc)
                    
                    self.log('info', f"Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, "
                                   f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                                   f"F1: {f1:.4f}, AUROC: {auroc:.4f}")
                    
                    # Add loss and accuracy to lists for plots
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_accuracies.append(np.mean(all_preds == all_labels))
                    val_accuracies.append(accuracy)
                    
                    # Save if there's improvement in loss or target metric
                    if val_loss <= best_val_loss - self.MINIMAL_IMPROVEMENT:
                        
                        best_val_loss = val_loss
                        patience_count = 0
                        
                        # Save checkpoint
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        checkpoint = {
                            'global_step': global_step,
                            'best_val_loss': best_val_loss,
                            'model': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'patience_count': patience_count,
                            'target_metric': target_metric,
                            'target_score': target_score,
                            'target_label': self.target_label
                        }
                        
                        checkpoint_name = f"ecgfm_best_finetuned_{self.target_label}.pt"
                        save_path = os.path.join(self.finetuned_dir, checkpoint_name)
                        
                        torch.save(checkpoint, save_path)
                        
                        self.log('info', f"New best val loss = {best_val_loss:.4f}. Checkpoint saved at step {global_step}")
                        
                        # Reduce regularization as reward
                        if self.dynamic_reg:
                            self.dynamic_regularizer(optimizer=optimizer, model=self.model, penalty=False)
                                    
                    elif target_score >= best_val_target_score + self.MINIMAL_IMPROVEMENT:
                        
                        best_val_target_score = target_score
                        
                        # Don't increment patience_count but don't reset it either
                        
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        checkpoint = {
                            'global_step': global_step,
                            'best_val_loss': best_val_loss,
                            'model': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'patience_count': patience_count,
                            'target_metric': target_metric,
                            'target_score': target_score,
                            'target_label': self.target_label
                        }
                        
                        checkpoint_name = f"ecgfm_best_finetuned_{self.target_label}_{target_metric}.pt"
                        save_path = os.path.join(self.finetuned_dir, checkpoint_name)
                        
                        torch.save(checkpoint, save_path)
                        
                        self.log('info', f"Val loss not improved but {target_metric} did (= {target_score:.4f}). "
                                       f"Checkpoint saved at step {global_step}")
                        
                        # Reduce regularization as reward
                        if self.dynamic_reg:
                            self.dynamic_regularizer(optimizer=optimizer, model=self.model, penalty=False)
                        
                    else:  # Loss not improved and target metric not improved
                        patience_count += 1
                        
                        if self.dynamic_reg and patience_count % (patience // 3) == 0 and patience_count < patience:
                            self.dynamic_regularizer(optimizer=optimizer, model=self.model, penalty=True)
                                    
                        if patience_count >= patience:
                            self.log('warning', f"Early stopping at step {global_step}.")
                            # Save plots before exiting
                            self._save_training_plots(train_losses, val_losses, train_accuracies, val_accuracies)
                            return
                    
                    # Return to training mode
                    self.model.train()
                    
                    # Reset batch train losses lists for next validation cycle
                    batch_train_losses = []
                
            # End of epoch
            self.log('info', f"Completed epoch {epoch+1}/{epochs}")
            
        self.log('info', f"End of training. Best val loss: {best_val_loss:.4f}, "
                       f"Best {target_metric}: {best_val_target_score:.4f}")
        
        # Save final plots
        self._save_training_plots(train_losses, val_losses, train_accuracies, val_accuracies)
    
    def _save_training_plots(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """
        Save training/validation plots for loss and accuracy
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_accuracies: List of training accuracies
            val_accuracies: List of validation accuracies
        """
        try:
            # Create destination directory for plots
            if self.output_dir:
                plot_dir = os.path.join(self.output_dir, self.target_label)
            else:
                plot_dir = os.path.join(self.finetuned_dir, "plots", self.target_label)
            os.makedirs(plot_dir, exist_ok=True)
            
            # Loss plot
            plt.figure(figsize=(10, 6))
            steps_range = range(1, len(train_losses) + 1)
            plt.plot(steps_range, train_losses, label='Training Loss')
            plt.plot(steps_range, val_losses, label='Validation Loss')
            plt.title(f'Training and Validation Losses for {self.target_label}')
            plt.xlabel('Validation Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            losses_plot_path = os.path.join(plot_dir, 'training_validation_losses.jpg')
            plt.savefig(losses_plot_path)
            plt.close()
            
            # Accuracy plot
            plt.figure(figsize=(10, 6))
            plt.plot(steps_range, train_accuracies, label='Training Accuracy', color='blue')
            plt.plot(steps_range, val_accuracies, label='Validation Accuracy', color='green')
            
            plt.title(f'Training and Validation Accuracy for {self.target_label}')
            plt.xlabel('Validation Steps')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            acc_plot_path = os.path.join(plot_dir, 'accuracy.jpg')
            plt.savefig(acc_plot_path)
            plt.close()
            
            self.log('info', f"Saved training/validation plots in {plot_dir}")
        except Exception as e:
            self.log('warning', f"Error creating or saving plots: {str(e)}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model on the test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Check that model and test data are ready
        if self.model is None:
            error_msg = "Model not initialized. Call build_model() first."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
            
        if self.test_loader is None:
            error_msg = "Test data not prepared. Call prepare_data() first."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
        
        self.log('info', f"Evaluating model on test set...")
        
        # Define output directory
        if self.output_dir:
            output_dir = os.path.join(self.output_dir, self.target_label)
        else:
            output_dir = os.path.join(self.finetuned_dir, "results", self.target_label)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.logger:
            self.logger.start_timer("evaluate_model")
        
        # Load best model if available
        best_model_path = os.path.join(self.finetuned_dir, f"ecgfm_best_finetuned_*.pt")
        
        # Find most recent file matching the pattern
        best_model_files = glob.glob(best_model_path)
        
        if best_model_files:
            # Sort by most recent modification time
            best_model_path = sorted(best_model_files, key=os.path.getmtime)[-1]
            self.log('info', f"Loading best model from {best_model_path}")
            try:
                # Try first with weights_only=True (safer)
                checkpoint = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model'])
            except Exception as e:
                self.log('warning', f"Error loading with weights_only=True: {e}")
                self.log('warning', "Attempting to load with weights_only=False...")
                # Fallback: use weights_only=False
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model'])
            
        
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        test_loss = 0.0
        all_preds_proba = []
        all_targets = []
        
        # Use BCEWithLogitsLoss for evaluation
        criterion = nn.BCEWithLogitsLoss()
        
        # Evaluate on test set
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Evaluation"):
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(source=data)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # Store predictions and targets
                all_preds_proba.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate average loss on test set
        avg_test_loss = test_loss / len(self.test_loader)
        
        # Convert to numpy arrays
        all_preds_proba = np.array(all_preds_proba)
        all_targets = np.array(all_targets)
        
        # Calculate metrics for binary classification
        all_preds = (all_preds_proba >= 0.5).astype(int)
        
        # Calculate basic metrics
        accuracy = np.mean(all_preds == all_targets)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        try:
            auroc = roc_auc_score(all_targets, all_preds_proba)
            auprc = average_precision_score(all_targets, all_preds_proba)
        except:
            auroc = 0.5
            auprc = np.mean(all_targets)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Calculate data for ROC and PR curves
        fpr, tpr, _ = roc_curve(all_targets, all_preds_proba)
        precision_vals, recall_vals, _ = precision_recall_curve(all_targets, all_preds_proba)
        
        # Collect metrics
        metrics = {
            'test_loss': avg_test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auroc': auroc,
            'auprc': auprc,
            'specificity': specificity,
            'npv': npv,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'confusion_matrix': cm,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'precision_vals': precision_vals,
            'recall_vals': recall_vals,
            'target_label': self.target_label
        }
        
        # Log confusion matrix
        self.log('debug', f"Confusion matrix: [[{tn}, {fp}], [{fn}, {tp}]]")
        
        # Create and save plots
        try:
            # ROC curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='red', label=f'AUROC = {auroc:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.target_label}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            auroc_path = os.path.join(output_dir, 'AUROC.jpg')
            plt.savefig(auroc_path)
            plt.close()
            self.log('debug', f"Saved ROC curve plot in {auroc_path}")
            
            # Precision-Recall curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, color='blue', label=f'AUPRC = {auprc:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {self.target_label}')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.tight_layout()
            auprc_path = os.path.join(output_dir, 'AUPRC.jpg')
            plt.savefig(auprc_path)
            plt.close()
            self.log('debug', f"Saved PR curve plot in {auprc_path}")
            
            # Confusion matrix plot
            plt.figure(figsize=(8, 8))
            sns_heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                      xticklabels=['Negative', 'Positive'],
                                      yticklabels=['Negative', 'Positive'])
            
            plt.title(f'Confusion Matrix - {self.target_label}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(output_dir, 'confusion_matrix.jpg')
            plt.savefig(cm_path)
            plt.close()
            self.log('debug', f"Saved confusion matrix plot in {cm_path}")
            
        except Exception as e:
            self.log('warning', f"Error creating plots: {str(e)}")
        
        # Save metrics to file
        try:
            # Save complete metrics as pickle
            metrics_df = pd.DataFrame([metrics])
            metrics_pkl_path = os.path.join(output_dir, 'metrics.pkl')
            metrics_df.to_pickle(metrics_pkl_path)
            self.log('debug', f"Saved pickle metrics in {metrics_pkl_path}")
            
            # Save simplified metrics as CSV
            metrics_simple = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
            metrics_csv_path = os.path.join(output_dir, 'metrics.csv')
            pd.DataFrame([metrics_simple]).to_csv(metrics_csv_path, index=False)
            self.log('debug', f"Saved CSV metrics in {metrics_csv_path}")
        except Exception as e:
            self.log('warning', f"Error saving metrics to file: {str(e)}")
        
        if self.logger:
            self.logger.stop_timer("evaluate_model")
            
            # Log detailed metrics
            self.logger.log_result(f"Model evaluation metrics", {
                "Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1-score": f"{f1:.4f}",
                "AUROC": f"{auroc:.4f}",
                "AUPRC": f"{auprc:.4f}",
                "Specificity": f"{specificity:.4f}",
                "Confusion matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]"
            })
        
        return metrics
    
    def configure_parameter_groups(self, learning_rate: float) -> List[Dict[str, Any]]:
        """
        Configure parameter groups with different learning rates for optimization.
        
        Args:
            learning_rate: Base learning rate to use
            
        Returns:
            List of dictionaries with parameters and their respective learning rates
        """
        parameters_group = []
        
        if self.layer_wise_lr:
            self.log('info', "Configuring learning rate by layer")
            
            # Feature extractor (if trainable)
            if hasattr(self.model, 'feature_extractor'):
                fe_params = [p for p in self.model.feature_extractor.parameters() if p.requires_grad]
                if fe_params:
                    parameters_group.append({
                        "params": fe_params,
                        "lr": learning_rate * 0.1  # Lower learning rate for feature extractor
                    })
                    self.log('info', f"Learning rate for feature extractor: {learning_rate * 0.1}")
            
            # Transformer layers
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                total_layers = len(self.model.encoder.layers)
                if self.transformer_blocks_to_unfreeze > 0:
                    # Divide layers to apply different learning rates
                    frozen_layers_idx = total_layers - self.transformer_blocks_to_unfreeze
                    
                    # Lower layers (closer to input) with lower learning rate
                    for i in range(frozen_layers_idx, total_layers - 2):
                        layer_params = [p for p in self.model.encoder.layers[i].parameters() if p.requires_grad]
                        if layer_params:
                            parameters_group.append({
                                "params": layer_params,
                                "lr": learning_rate * 0.5
                            })
                    
                    # Top layers with full learning rate
                    for i in range(total_layers - 2, total_layers):
                        layer_params = [p for p in self.model.encoder.layers[i].parameters() if p.requires_grad]
                        if layer_params:
                            parameters_group.append({
                                "params": layer_params,
                                "lr": learning_rate
                            })
            
            # Other encoder layers (layer norm, etc.)
            encoder_params = [
                p for n, p in self.model.named_parameters() 
                if p.requires_grad and 'encoder' in n and 'layers' not in n
            ]
            if encoder_params:
                parameters_group.append({
                    "params": encoder_params,
                    "lr": learning_rate * 0.5
                })
            
            # Classification layer always with higher learning rate
            classifier_params = []
            if hasattr(self.model, 'classification_head'):
                classifier_params = list(self.model.classification_head.parameters())
            
            if classifier_params:
                parameters_group.append({
                    "params": classifier_params,
                    "lr": learning_rate * 2.0
                })
                self.log('info', f"Learning rate for classification_head: {learning_rate * 2.0}")
            
            # Other uncategorized parameters
            other_params = [
                p for n, p in self.model.named_parameters() 
                if p.requires_grad and 'encoder' not in n and 'feature_extractor' not in n 
                and 'classification_head' not in n
            ]
            
            if other_params:
                parameters_group.append({
                    "params": other_params,
                    "lr": learning_rate
                })
        else:
            # Use uniform learning rate for all trainable parameters
            parameters_group.append({
                "params": filter(lambda p: p.requires_grad, self.model.parameters()),
                "lr": learning_rate
            })
        
        return parameters_group
    
    
if __name__ == "__main__":      
    # Crea modello ECG-FM
    model_config = Wav2Vec2CMSCConfigCustomFtCode15()
    model = Wav2Vec2CMSCModelCustomFtCode15(model_config)
    
    for name, param in model.named_parameters():
        print(name)