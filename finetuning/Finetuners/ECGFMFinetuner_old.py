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
from typing import Dict, Any
import seaborn as sns

from Finetuners.BaseFinetuner import BaseFinetuner
from Finetuners.DataSampler import DataSampler
from Finetuners.tools.wave2vec2_cmsc_custom_ft_code15 import Wav2Vec2CMSCConfigCustomFtCode15, Wav2Vec2CMSCModelCustomFtCode15


class ECGDataset(torch.utils.data.Dataset):
    """
    Custom dataset class to handle ECG data for PyTorch models.
    """
    
    def __init__(self, features, labels):
        """
        Initialize the dataset with features and labels.
        
        Args:
            features: DataFrame with ECG features
            labels: Series with binary labels
        """
        # Extract the tensors
        tensor_list = []
        for i in range(len(features)):
            # Extract tensor from each row
            series = features.iloc[i] 
            tensor_list.append(series[0])

        # Stack tensors into a single tensor of shape [N, 12, 2500]
        self.features = torch.stack(tensor_list)
        
        # Convert labels to tensor
        if isinstance(labels, pd.Series):
            self.labels = torch.tensor(labels.values, dtype=torch.float32).reshape(-1, 1)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        
        # Print shapes for debugging
        print(f"Dataset initialized - Features: {self.features.shape}, Labels: {self.labels.shape}")
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Return the features and label for a given index.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple (features, label) for the specified index
        """
        return self.features[idx], self.labels[idx]


class ECGFMFinetuner(BaseFinetuner):
    """
    ECG-FM specific finetuner for ECG data.
    Implements the methods of the BaseFinetuner abstract class.
    """
    
    def __init__(self,
             processed_data_dir: str,
             ecg_dataset_path: str,
             meta_dataset_path: str,
             checkpoint_path: str,
             save_dir: str = "checkpoints",
             dataset_name: str = "code15",
             logger=None,
             output_dir=None,  # Add output_dir parameter
             **kwargs):
        """
        Initialize the ECG-FM finetuner with the provided parameters.
        
        Args:
            processed_data_dir: Directory containing processed data
            ecg_dataset_path: Path to the ECG dataset
            meta_dataset_path: Path to the metadata dataset
            checkpoint_path: Path to the pretrained model checkpoint
            save_dir: Directory to save finetuned model checkpoints
            dataset_name: Name of the dataset (default: "code15")
            logger: Logger instance for tracking progress
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
        # Get the batch_size from finetuning_params in kwargs, or use a default
        # This ensures the batch_size from config.yaml is used consistently
        self.batch_size = kwargs.get('finetuning_params', {}).get('batch_size', 32)
        self.output_dir = output_dir  # Store output_dir
        
        # Dataset-specific paths
        self.finetuned_dir = os.path.join(self.save_dir, f'{dataset_name}_ecgfm_finetuned')
        os.makedirs(self.finetuned_dir, exist_ok=True)
        
        # Log ECG-FM specific parameters
        self.log('debug', f"Dataset name: {self.dataset_name}")
        self.log('debug', f"Batch size: {self.batch_size}")
        self.log('debug', f"Finetuned directory: {self.finetuned_dir}")
        self.log('debug', f"Output directory: {self.output_dir}")
    
    def build_model(self) -> None:
        """
        Build the ECG-FM model from checkpoint for finetuning.
        """
        self.log('info', f"Loading ECG-FM model from checkpoint: {self.checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create ECG-FM model
            model_config = Wav2Vec2CMSCConfigCustomFtCode15()
            self.model = Wav2Vec2CMSCModelCustomFtCode15(model_config)
            
            # Log model architecture summary
            self.log('debug', f"Model architecture: {type(self.model).__name__}")
          
            # Load checkpoint weights
            self.model.load_state_dict(checkpoint['model'], strict=False)
            
            # Move model to device
            self.model.to(self.device)
            
            # Calculate model size
            model_size_mb = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)  # Assuming float32 (4 bytes)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            self.log('info', f"ECG-FM model loaded successfully. Model size: {model_size_mb:.2f} MB")
            self.log('info', f"Total parameters: {total_params:,}")
            self.log('info', f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
            # Log memory usage after model loading
            if self.logger:
                self.logger.log_memory_usage()
                
        except Exception as e:
            error_msg = f"Failed to load ECG-FM model: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
    
    def prepare_data(self, target_label: str, batch_size: int = None) -> None:
        """
        Prepare the data for finetuning by creating train/val/test splits.
        
        Args:
            target_label: Target label for binary classification
            batch_size: Batch size for training, if None use the one from config
        """
        self.target_label = target_label
        # Use the provided batch_size parameter if given, otherwise use the one from class init
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        # Initialize DataSampler with config from kwargs
        sampler_config = self.kwargs.get('data_sampler_params', {})
        self.log('info', f"Initializing DataSampler with target label: {target_label}")
        
        try:
            self.data_sampler = DataSampler(sampler_config)
            
            # Check if target label is available
            available_labels = self.data_sampler.get_all_target_labels()
            if target_label not in available_labels:
                error_msg = f"Target label '{target_label}' not found in available labels: {available_labels}"
                self.log('error', error_msg)
                raise ValueError(error_msg)
                
            # Create train/val/test splits
            self.log('info', f"Creating dataset splits for label: {target_label}")
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_sampler.create_train_val_test_split(target_label)
            
            self.log('info', f"Got splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples")
            
            # Create PyTorch datasets
            train_dataset = ECGDataset(X_train, y_train)
            val_dataset = ECGDataset(X_val, y_val)
            test_dataset = ECGDataset(X_test, y_test)
            
            num_workers = self.kwargs.get('finetuning_params', {}).get('num_workers', 4)
            pin_memory = self.kwargs.get('finetuning_params', {}).get('pin_memory', True)
            prefetch_factor = self.kwargs.get('finetuning_params', {}).get('prefetch_factor', 2)
            
            self.log('info', f"Usando batch size: {batch_size}, workers: {num_workers}, "
                            f"pin_memory: {pin_memory}, prefetch_factor: {prefetch_factor}")
            
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
            
            self.log('info', f"Creati data loader - Training: {len(self.train_loader)} batch, "
                        f"Validation: {len(self.val_loader)} batch, Test: {len(self.test_loader)} batch")
        
        except Exception as e:
            error_msg = f"Error preparing data: {str(e)}"
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
    
    def finetune(self, epochs: int, learning_rate: float) -> None:
        """
        Finetune il modello sui dati preparati con ottimizzazioni per le performance.
        
        Args:
            epochs: Numero massimo di epoche di training
            learning_rate: Learning rate per l'ottimizzazione
        """
        # Verifica che il modello e i dati siano pronti
        if self.model is None:
            error_msg = "Modello non inizializzato. Chiama build_model() prima."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
            
        if self.train_loader is None or self.val_loader is None:
            error_msg = "Dati non preparati. Chiama prepare_data() prima."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
        
        # OTTIMIZZAZIONE 1: Abilita cudnn benchmark per migliorare le performance
        torch.backends.cudnn.benchmark = True
        
        # OTTIMIZZAZIONE 2: Configura il freezing dei layer dal file di configurazione
        layers_to_freeze = self.kwargs.get('finetuning_params', {}).get('layers_to_freeze', 0)
        if layers_to_freeze > 0:
            # Congela i primi N layer
            n_frozen = 0
            for name, param in self.model.named_parameters():
                if n_frozen < layers_to_freeze:
                    param.requires_grad = False
                    n_frozen += 1
                else:
                    param.requires_grad = True
            self.log('info', f"Freezing {layers_to_freeze} layers for faster training")
        
        
        # Freeze all convolutional layers
        for module in self.model.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                for param in module.parameters():
                    param.requires_grad = False
                self.log('info', 'Frozen all convolutional layers for fine-tuning')
        
        # OTTIMIZZAZIONE 3: Multi-GPU training se disponibile
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.log('info', f"Found {num_gpus} GPUs! Utilizzando DataParallel su {min(2, num_gpus)} GPU")
            # Limita a 2 GPU come richiesto
            self.model = nn.DataParallel(self.model, device_ids=list(range(min(2, num_gpus))))
        
        # Parametri di early stopping
        patience = self.kwargs.get('finetuning_params', {}).get('patience', 5)
        epochs_without_improvement = 0
        best_val_accuracy = 0.0
        
        self.log('info', f"Avvio finetuning con {epochs} epoche, learning rate {learning_rate}, e patience {patience}")
        
        # OTTIMIZZAZIONE 4: Utilizza AdamW invece di Adam
        beta1 = self.kwargs.get('finetuning_params', {}).get('beta1', 0.9)
        beta2 = self.kwargs.get('finetuning_params', {}).get('beta2', 0.999)
        weight_decay = self.kwargs.get('finetuning_params', {}).get('weight_decay', 0.01)
        self.log('info', f"Uso AdamW con beta1: {beta1}, beta2: {beta2}, weight_decay: {weight_decay}")
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        # OTTIMIZZAZIONE 5: Gradient accumulation per simulare batch più grandi
        accumulation_steps = self.kwargs.get('finetuning_params', {}).get('accumulation_steps', 1)
        self.log('info', f"Utilizzo accumulation steps: {accumulation_steps}")
        
        # OTTIMIZZAZIONE 6: Learning rate scheduler (OneCycleLR)
        # (Removed scheduler as per instructions)
        
        
        # Liste per memorizzare le loss e le accuracy per i grafici
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Training/validation loop
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(epochs):
            # Fase di training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Ottimizza il dataloader
            progress_bar = tqdm(self.train_loader, desc=f"Epoca {epoch+1}/{epochs}")

            # Inizializza il gradiente all'inizio di ogni epoca (fuori dal loop batch)
            optimizer.zero_grad()

            for batch_idx, (data, targets) in enumerate(progress_bar):
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass senza mixed precision
                outputs = self.model(source=data)
                loss = criterion(outputs, targets) / accumulation_steps

                # Backward pass senza scaling
                loss.backward()

                # Aggiorna il modello ogni accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # taglia i gradienti oltre una certa norma
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                # Aggiorna le statistiche
                train_loss += loss.item() * accumulation_steps

                # Calcola l'accuracy del training
                with torch.no_grad():
                    train_preds = (torch.sigmoid(outputs) >= 0.5).float()
                    train_total += targets.size(0)
                    train_correct += (train_preds == targets).sum().item()

                # Aggiorna la barra di progresso
                progress_bar.set_postfix({
                    "train_loss": train_loss / (batch_idx + 1)
                })

            # Gestisci il caso in cui il numero di batch non è divisibile per accumulation_steps
            if (batch_idx + 1) % accumulation_steps != 0:
                # taglia i gradienti oltre una certa norma
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                    
            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = train_correct / train_total
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            self.log('info', f"Epoca {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, "
                    f"Training Accuracy: {train_accuracy:.4f}")
            
            # Fase di validazione
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in self.val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(source=data, mask=False)
                    
                    # Calcola loss
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calcola accuracy
                    predictions = (torch.sigmoid(outputs) >= 0.5).float()
                    total += targets.size(0)
                    correct += (predictions == targets).sum().item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = correct / total
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            self.log('info', f"Epoca {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}, "
                    f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_epoch = epoch
                
                # Salva il miglior modello (rimuovi DataParallel se presente)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                checkpoint_path = os.path.join(self.finetuned_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'accuracy': best_val_accuracy,
                    'target_label': self.target_label,
                }, checkpoint_path)
                
                self.log('info', f"Salvato il miglior modello con validation accuracy {best_val_accuracy:.4f} in {checkpoint_path}")
            else:
                epochs_without_improvement += 1
                self.log('info', f"Nessun miglioramento dell'accuracy di validazione. "
                        f"Epoche senza miglioramento: {epochs_without_improvement}/{patience}")
                
                # Controlla la condizione di early stopping
                if epochs_without_improvement >= patience:
                    self.log('info', f"Early stopping attivato dopo {epoch+1} epoche. "
                            f"Miglior validation accuracy: {best_val_accuracy:.4f} all'epoca {best_epoch+1}")
                    break

        # Crea e salva i grafici di loss e accuracy
        try:
            # Crea la directory di destinazione per i grafici
            plot_dir = os.path.join(self.finetuned_dir, {self.target_label})
            os.makedirs(plot_dir, exist_ok=True)
            
            # Grafico delle loss di training e validation
            plt.figure(figsize=(10, 6))
            epochs_range = range(1, len(train_losses) + 1)
            plt.plot(epochs_range, train_losses, label='Training Loss')
            plt.plot(epochs_range, val_losses, label='Validation Loss')
            plt.title(f'Training e Validation Losses per {self.target_label}')
            plt.xlabel('Epoche')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            losses_plot_path = os.path.join(plot_dir, 'training_validation_losses.jpg')
            plt.savefig(losses_plot_path)
            plt.close()
            
            # Grafico delle accuracy di training e validation
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_range, train_accuracies, label='Training Accuracy', color='blue')
            plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='green')
            
            plt.title(f'Training e Validation Accuracy per {self.target_label}')
            plt.xlabel('Epoche')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            acc_plot_path = os.path.join(plot_dir, 'accuracy.jpg')
            plt.savefig(acc_plot_path)
            plt.close()
            
            self.log('info', f"Salvati i grafici di training/validation in {plot_dir}")
        except Exception as e:
            self.log('warning', f"Errore nella creazione o salvataggio dei grafici: {str(e)}")

        final_message = "Early stopping attivato" if epochs_without_improvement >= patience else "Training completato"
        self.log('info', f"{final_message} con miglior validation accuracy: {best_val_accuracy:.4f}")
    
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the finetuned model on the test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if model and test data are ready
        if self.model is None:
            error_msg = "Model not initialized. Call build_model() first."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
            
        if self.test_loader is None:
            error_msg = "Test data not prepared. Call prepare_data() first."
            self.log('error', error_msg)
            raise RuntimeError(error_msg)
        
        self.log('info', f"Evaluating model on test set...")
        
        # Define output directory - use the provided output_dir if available
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.start_timer("evaluate_model")
        
        # Load best model if available
        best_model_path = os.path.join(self.finetuned_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.log('info', f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            if hasattr(self.model, 'module') and not any(k.startswith('module.') for k in state_dict.keys()):
                self.log('info', "Aggiungendo prefisso 'module.' alle chiavi dello state_dict per compatibilità con DataParallel")
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_state_dict[f'module.{key}'] = value
                state_dict = new_state_dict
        
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
            for data, targets in tqdm(self.test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(source=data, mask=False)  # Raw logits
                
                # Calculate loss
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # Store predictions and targets
                all_preds_proba.extend(torch.sigmoid(outputs).cpu().numpy())  # Apply sigmoid here
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate average test loss
        avg_test_loss = test_loss / len(self.test_loader)
        
        # Convert to numpy arrays
        all_preds_proba = np.array(all_preds_proba)
        all_targets = np.array(all_targets)
        
        # Calculate binary classification metrics
        all_preds = (all_preds_proba >= 0.5).astype(int)
        
        # Calculate basic metrics
        accuracy = np.mean(all_preds == all_targets)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        auroc = roc_auc_score(all_targets, all_preds_proba)
        auprc = average_precision_score(all_targets, all_preds_proba)
        
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
        self.log('debug', f"Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")
        
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
            auroc_path = os.path.join(self.output_dir, 'AUROC.jpg')
            plt.savefig(auroc_path)
            plt.close()
            self.log('debug', f"Saved ROC curve plot to {auroc_path}")
            
            # Precision-Recall curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, color='blue', label=f'AUPRC = {auprc:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {self.target_label}')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.tight_layout()
            auprc_path = os.path.join(self.output_dir, 'AUPRC.jpg')
            plt.savefig(auprc_path)
            plt.close()
            self.log('debug', f"Saved PR curve plot to {auprc_path}")
            
            # Confusion matrix plot
            plt.figure(figsize=(8, 8))
            sns_heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                        xticklabels=['Negative', 'Positive'],
                                        yticklabels=['Negative', 'Positive'])
            
                        
            plt.title(f'Confusion Matrix - {self.target_label}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(self.output_dir, 'confusion_matrix.jpg')
            plt.savefig(cm_path)
            plt.close()
            self.log('debug', f"Saved confusion matrix plot to {cm_path}")
            
        except Exception as e:
            self.log('warning', f"Error creating plots: {str(e)}")
        
        # Save metrics to files
        try:
            # Save full metrics as pickle
            metrics_df = pd.DataFrame([metrics])
            metrics_pkl_path = os.path.join(self.output_dir, 'metrics.pkl')
            metrics_df.to_pickle(metrics_pkl_path)
            self.log('debug', f"Saved metrics pickle to {metrics_pkl_path}")
            
            # Save simplified metrics as CSV
            metrics_simple = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
            metrics_csv_path = os.path.join(self.output_dir, 'metrics.csv')
            pd.DataFrame([metrics_simple]).to_csv(metrics_csv_path, index=False)
            self.log('debug', f"Saved metrics CSV to {metrics_csv_path}")
        except Exception as e:
            self.log('warning', f"Error saving metrics to files: {str(e)}")
        
        if self.logger:
            self.logger.stop_timer("evaluate_model")
            
            # Log detailed metrics
            self.logger.log_result(f"Model Evaluation Metrics", {
                "Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1-score": f"{f1:.4f}",
                "AUROC": f"{auroc:.4f}",
                "AUPRC": f"{auprc:.4f}",
                "Specificity": f"{specificity:.4f}",
                "Confusion Matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]"
            })
        
        return metrics