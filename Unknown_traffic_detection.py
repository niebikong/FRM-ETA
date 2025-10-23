import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

from bin.frm_eta_ood import FRMETAOOD
from lib import Transformations, build_dataset, prepare_tensors, DATA
from util import metrics


class FRMETAOODDetector:
    """Out-of-distribution detector based on FRM-ETA"""

    def __init__(self, model_path, dataset_name='dapp', num_known_classes=None, config_path=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.num_known_classes = num_known_classes
        self.config_path = config_path
        self.model = None
        self.dataset = None
        self.model_config = None

        self.dataset_paths = {
            'dapp': r'data/dapp_84_features_label.csv',
            'mal': r'data/label_encodered_malicious_TLS-1.csv',
        }
        
    def load_model_config(self):
        """Load model configuration"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.model_config = config.get('model_config', {})
                print(f"Model config loaded from {self.config_path}")
        else:
            self.model_config = {
                'n_heads': 12,
                'd_token': 192,
                'n_layers': 5,
                'ffn_dropout': 0.1,
                'attention_dropout': 0.4,
                'd_ffn_factor': 2.0,
                'residual_dropout': 0.1
            }
            print("Using default model config")

    def load_model_and_data(self):
        """Load model and dataset"""
        print("Loading model and dataset...")

        self.load_model_config()

        # Determine dataset directory name
        if self.num_known_classes:
            dataset_dir_name = f"{self.dataset_name}_ood_{self.num_known_classes}classes"
        else:
            # Try to infer from existing directories
            possible_dirs = [d for d in os.listdir('data') if d.startswith(f"{self.dataset_name}_ood_")]
            if possible_dirs:
                dataset_dir_name = possible_dirs[0]
                # Extract number of classes from directory name
                self.num_known_classes = int(dataset_dir_name.split('_')[-1].replace('classes', ''))
                print(f"Inferred dataset directory: {dataset_dir_name}")
            else:
                dataset_dir_name = self.dataset_name
                print(f"Using original dataset: {dataset_dir_name}")

        # Load dataset
        normalization = 'quantile'
        transformation = Transformations(normalization=normalization)
        self.dataset = build_dataset(DATA / dataset_dir_name, transformation, False)  # Disable caching

        # Create model
        n_num_features = self.dataset.n_num_features
        cardinalities = self.dataset.get_category_sizes('train')
        n_categories = len(cardinalities)
        cardinalities = None if n_categories == 0 else cardinalities
        d_out = self.dataset.n_classes or 1

        model_config = {
            'd_numerical': n_num_features,
            'categories': cardinalities,
            'd_out': d_out,
            'activation': 'reglu',
            'initialization': 'kaiming',
            'n_heads': self.model_config.get('n_heads', 12),  
            'prenormalization': True,
            'residual_dropout': self.model_config.get('residual_dropout', 0.1),  
            'attention_dropout': self.model_config.get('attention_dropout', 0.4),
            'd_ffn_factor': self.model_config.get('d_ffn_factor', 2.0),  
            'd_token': self.model_config.get('d_token', 192),  
            'ffn_dropout': self.model_config.get('ffn_dropout', 0.1),
            'n_layers': self.model_config.get('n_layers', 5),  
            'token_bias': True,
            'kv_compression': None,
            'kv_compression_sharing': None
        }

        self.model = FRMETAOOD(**model_config).to(self.device)

        # Load pre-trained weights
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Warning: Model file {self.model_path} not found!")

        self.model.eval()

        print(f"Dataset: {dataset_dir_name}")
        print(f"Features: {n_num_features}, Classes: {d_out}")
        print(f"Model config: {model_config}")
        
    def extract_features_reference_style(self, data_loader):
        """
        Extract features following reference_code approach:
        1. Use feature_list to get features from each layer
        2. Apply global average pooling to each layer (for FRM-ETA, features are already CLS tokens)
        3. Concatenate features from all layers
        """
        all_features = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for batch in data_loader:
                x_num, x_cat, y = (batch[0], None, batch[1]) if len(batch) == 2 else batch

                # Use feature_list method to get features from each layer
                logits, feature_list = self.model.feature_list(x_num, x_cat)

                # Apply global average pooling to each feature layer
                # Each element in feature_list is already [batch_size, d_token] CLS token features
                pooled_features = feature_list

                # Concatenate all pooled features
                # FRM-ETA: 128 + 128 + 128 + 128 = 512
                concatenated_features = torch.cat(pooled_features, dim=1)

                all_features.append(concatenated_features.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_scores.append(logits.cpu().numpy())

        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        all_scores = np.vstack(all_scores)

        return all_features, all_labels, all_scores
    
    def prepare_data_loaders(self):
        if self.dataset.X_num['train'].dtype == np.float64:
            print("Converting data to float32 for mixed precision compatibility...")
            self.dataset.X_num = {k: v.astype(np.float32) for k, v in self.dataset.X_num.items()}

        X_num, X_cat, ys = prepare_tensors(self.dataset, device=self.device)

        batch_size = 256

        data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
        train_dataset = TensorDataset(*(d['train'] for d in data_list))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(*(d['val'] for d in data_list))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(*(d['test'] for d in data_list))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        dataset_dir = self.dataset_name
        if self.num_known_classes:
            dataset_dir = f"{self.dataset_name}_ood_{self.num_known_classes}classes"

        unknown_data_path = DATA / dataset_dir / 'X_unknown.npy'
        unknown_labels_path = DATA / dataset_dir / 'y_unknown.npy'

        if os.path.exists(unknown_data_path) and os.path.exists(unknown_labels_path):
            unknown_data = np.load(unknown_data_path)
            unknown_labels = np.load(unknown_labels_path)

            if unknown_data.dtype == np.float64:
                unknown_data = unknown_data.astype(np.float32)

            unknown_tensor = torch.tensor(unknown_data, dtype=torch.float32).to(self.device)
            unknown_labels_tensor = torch.tensor(unknown_labels, dtype=torch.long).to(self.device)

            unknown_dataset = TensorDataset(unknown_tensor, unknown_labels_tensor)
            unknown_loader = DataLoader(unknown_dataset, batch_size=batch_size, shuffle=False)

            print(f"Unknown data loaded: {unknown_data.shape}")
        else:
            print(f"Warning: Unknown data not found at {unknown_data_path}")
            unknown_loader = None

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'unknown': unknown_loader
        }
    
    def compute_anomaly_scores(self, train_features, train_labels, test_features):

        print("Computing OOD scores...")
        mean_feat = train_features.mean(0)
        std_feat = train_features.std(0)

        def prepos_feat_ssd(x):
            return (x - mean_feat) / (std_feat + 1e-10)

        ftrain_ssd = prepos_feat_ssd(train_features)
        ftest_ssd = prepos_feat_ssd(test_features)

        class_num = len(np.unique(train_labels))
        print(f"Number of classes: {class_num}")

        inv_sigma_cls = [None for _ in range(class_num)]
        mean_cls = [None for _ in range(class_num)]

        cov = lambda x: np.cov(x.T, bias=True)

        # Compute mean and inverse covariance matrix for each class
        for cls in range(class_num):
            # Get current class features
            cls_mask = train_labels == cls
            cls_features = ftrain_ssd[cls_mask]

            if len(cls_features) > 1:  # Ensure enough samples
                # Compute class mean
                mean_cls[cls] = cls_features.mean(0)

                # Center features
                feat_cls_center = cls_features - mean_cls[cls]

                # Compute inverse covariance matrix
                try:
                    cov_matrix = cov(feat_cls_center)
                    inv_sigma_cls[cls] = np.linalg.pinv(cov_matrix)
                except:
                    # If calculation fails, use identity matrix
                    inv_sigma_cls[cls] = np.eye(ftrain_ssd.shape[1])
            else:
                # Insufficient samples
                mean_cls[cls] = ftrain_ssd[cls_mask].mean(0) if np.sum(cls_mask) > 0 else np.zeros(ftrain_ssd.shape[1])
                inv_sigma_cls[cls] = np.eye(ftrain_ssd.shape[1])


        def calculate_distance_scores(X):
            """Calculate distance-based scores"""
            score_cls = np.zeros((class_num, len(X)))

            for cls in range(class_num):
                if inv_sigma_cls[cls] is not None and mean_cls[cls] is not None:
                    inv_sigma = inv_sigma_cls[cls]
                    mean = mean_cls[cls]

                    # Compute difference
                    z = X - mean

                    # Compute Mahalanobis distance: -0.5 * (x-μ)^T * Σ^(-1) * (x-μ)
                    # Here use negative sign, so that the closer the distance, the higher the score (more like the class)
                    score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)

            # Return maximum score for each sample across all classes
            return score_cls.max(0)

        ssd_scores = calculate_distance_scores(ftest_ssd)

        return ssd_scores

    def compute_threshold_based_metrics(self, train_anomaly_scores, id_anomaly_scores, ood_anomaly_scores):
        """
        Compute threshold-based classification metrics
        Threshold based on training set features, so that 95% of the training data are correctly classified as ID
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        print("\n" + "="*70)
        print("THRESHOLD-BASED CLASSIFICATION METRICS (95% Train ID Correct)")
        print("="*70)

        # Merge test data
        all_scores = np.concatenate([id_anomaly_scores, ood_anomaly_scores])
        all_labels = np.concatenate([np.zeros(len(id_anomaly_scores)), np.ones(len(ood_anomaly_scores))])

        print(f"Total test samples: {len(all_scores)} (ID: {len(id_anomaly_scores)}, OOD: {len(ood_anomaly_scores)})")
        print(f"Train samples for threshold: {len(train_anomaly_scores)}")

        # Set threshold based on training set, so that 95% of the training data are correctly classified as ID
        train_scores_sorted = np.sort(train_anomaly_scores)
        threshold_95 = train_scores_sorted[int(0.95 * len(train_anomaly_scores))]

        print(f"Threshold (95% train ID correct): {threshold_95:.4f}")
        print(f"Train anomaly scores - Mean: {train_anomaly_scores.mean():.4f}, Std: {train_anomaly_scores.std():.4f}")

        # Predict based on threshold (score > threshold predict as OOD)
        y_pred = (all_scores > threshold_95).astype(int)

        # Compute classification metrics
        accuracy = accuracy_score(all_labels, y_pred)
        precision = precision_score(all_labels, y_pred, zero_division=0)
        recall = recall_score(all_labels, y_pred, zero_division=0)
        f1 = f1_score(all_labels, y_pred, zero_division=0)

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Compute class accuracies
        id_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0  # ID samples correctly classified as ID
        ood_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0  # OOD samples correctly classified as OOD

        print(f"\nClassification Results:")
        print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision (OOD):  {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall (OOD):     {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score (OOD):   {f1:.4f} ({f1*100:.2f}%)")
        print(f"  ID Accuracy:      {id_accuracy:.4f} ({id_accuracy*100:.2f}%)")
        print(f"  OOD Accuracy:     {ood_accuracy:.4f} ({ood_accuracy*100:.2f}%)")

        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 ID    OOD")
        print(f"Actual    ID   {tn:5d}  {fp:5d}")
        print(f"          OOD  {fn:5d}  {tp:5d}")

        # Save confusion matrix to file
        self.save_evaluation_metrics(cm, tn, fp, fn, tp, accuracy, precision, recall, f1, id_accuracy, ood_accuracy, threshold_95)

        return {
            'threshold': threshold_95,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'id_accuracy': id_accuracy,
            'ood_accuracy': ood_accuracy
        }

    def save_evaluation_metrics(self, cm, tn, fp, fn, tp, accuracy, precision, recall, f1, id_accuracy, ood_accuracy, threshold):
        """Save evaluation metrics and confusion matrix to file"""
        # Determine save path
        model_dir = Path(self.model_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create confusion matrix filename
        cm_filename = f"confusion_matrix_{self.dataset_name}_{self.num_known_classes}classes_{timestamp}.txt"
        cm_filepath = model_dir / cm_filename

        # Prepare save content
        content = []
        content.append("=" * 80)
        content.append("OOD DETECTION CONFUSION MATRIX AND METRICS")
        content.append("=" * 80)
        content.append(f"Dataset: {self.dataset_name}")
        content.append(f"Known Classes: {self.num_known_classes}")
        content.append(f"Model Path: {self.model_path}")
        content.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")

        content.append("THRESHOLD-BASED CLASSIFICATION METRICS:")
        content.append(f"  Threshold (95% train ID correct): {threshold:.4f}")
        content.append(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        content.append(f"  Precision (OOD):  {precision:.4f} ({precision*100:.2f}%)")
        content.append(f"  Recall (OOD):     {recall:.4f} ({recall*100:.2f}%)")
        content.append(f"  F1-Score (OOD):   {f1:.4f} ({f1*100:.2f}%)")
        content.append(f"  ID Accuracy:      {id_accuracy:.4f} ({id_accuracy*100:.2f}%)")
        content.append(f"  OOD Accuracy:     {ood_accuracy:.4f} ({ood_accuracy*100:.2f}%)")
        content.append("")

        content.append("CONFUSION MATRIX:")
        content.append("                 Predicted")
        content.append("                 ID    OOD")
        content.append(f"Actual    ID   {tn:5d}  {fp:5d}")
        content.append(f"          OOD  {fn:5d}  {tp:5d}")
        content.append("")

        content.append("CONFUSION MATRIX DETAILS:")
        content.append(f"  True Negatives (TN):  {tn:5d}  (ID correctly classified as ID)")
        content.append(f"  False Positives (FP): {fp:5d}  (ID incorrectly classified as OOD)")
        content.append(f"  False Negatives (FN): {fn:5d}  (OOD incorrectly classified as ID)")
        content.append(f"  True Positives (TP):  {tp:5d}  (OOD correctly classified as OOD)")
        content.append("")

        content.append("NUMPY CONFUSION MATRIX:")
        content.append(f"[[{cm[0,0]:5d} {cm[0,1]:5d}]")
        content.append(f" [{cm[1,0]:5d} {cm[1,1]:5d}]]")
        content.append("")
        content.append("=" * 80)

        # Write to file
        with open(cm_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

        print(f"\nConfusion matrix saved to: {cm_filepath}")
        return cm_filepath

    def save_detection_scores(self, known_anomaly_scores, unknown_anomaly_scores):
        """Save detection scores to NPZ file"""
        # Determine save path
        model_dir = Path(self.model_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create anomaly scores filename
        scores_filename = f"anomaly_scores_{self.dataset_name}_{self.num_known_classes}classes_{timestamp}.npz"
        scores_filepath = model_dir / scores_filename

        # Save anomaly scores
        np.savez_compressed(
            scores_filepath,
            known_anomaly_scores=known_anomaly_scores,
            unknown_anomaly_scores=unknown_anomaly_scores,
            dataset_name=self.dataset_name,
            num_known_classes=self.num_known_classes,
            model_path=str(self.model_path),
            timestamp=timestamp
        )

        print(f"\nAnomaly scores saved to: {scores_filepath}")
        print(f"   - Known (ID) scores shape: {known_anomaly_scores.shape}")
        print(f"   - Unknown (OOD) scores shape: {unknown_anomaly_scores.shape}")

        return scores_filepath

    def extract_second_last_layer_features(self, data_loader, label_value):
        """
        Extract model second-to-last layer features

        Args:
            data_loader: data loader
            label_value: label value (0 for known/ID, 1 for unknown/OOD)

        Returns:
            features: second-to-last layer features [N, d_token]
            labels: corresponding labels [N] (all label_value)
        """
        all_features = []

        with torch.no_grad():
            for batch in data_loader:
                x_num, x_cat, y = (batch[0], None, batch[1]) if len(batch) == 2 else batch

                # Use feature_list to get all layer features
                logits, feature_list = self.model.feature_list(x_num, x_cat)

                # Get second-to-last layer features (feature_list[-2])
                # feature_list contains CLS token features for each layer
                if len(feature_list) >= 2:
                    second_last_features = feature_list[-2]  # second-to-last layer
                else:
                    # If not enough layers, use last layer
                    second_last_features = feature_list[-1]

                all_features.append(second_last_features.cpu().numpy())

        # Merge all features
        all_features = np.vstack(all_features)

        # Create labels (all label_value)
        labels = np.full(all_features.shape[0], label_value, dtype=np.int32)

        return all_features, labels

    def save_extracted_features(self):
        """
        Extract and save model features to NPZ file
        Known class traffic labeled as 0, unknown class traffic labeled as 1
        """
        print("\nExtracting second-to-last layer features...")

        # Prepare data loaders
        data_loaders = self.prepare_data_loaders()

        # Check if unknown data exists
        if data_loaders['unknown'] is None:
            print("No unknown data available for feature extraction!")
            return None

        # Extract known class features (label=0)
        print("Extracting features from test set (known/ID)...")
        known_features, known_labels = self.extract_second_last_layer_features(
            data_loaders['test'], label_value=0
        )

        print("Extracting features from unknown set (OOD)...")
        unknown_features, unknown_labels = self.extract_second_last_layer_features(
            data_loaders['unknown'], label_value=1
        )

        # Merge features and labels
        all_features = np.vstack([known_features, unknown_features])
        all_labels = np.concatenate([known_labels, unknown_labels])

        print(f"Known features shape: {known_features.shape}")
        print(f"Unknown features shape: {unknown_features.shape}")
        print(f"Total features shape: {all_features.shape}")
        print(f"Labels shape: {all_labels.shape}")
        print(f"Label distribution - Known (0): {np.sum(all_labels == 0)}, Unknown (1): {np.sum(all_labels == 1)}")

        # Determine save path
        model_dir = Path(self.model_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create feature filename
        features_filename = f"second_last_layer_features_{self.dataset_name}_{self.num_known_classes}classes_{timestamp}.npz"
        features_filepath = model_dir / features_filename

        # Save features and labels
        np.savez_compressed(
            features_filepath,
            features=all_features,
            labels=all_labels,
            known_features=known_features,
            unknown_features=unknown_features,
            known_labels=known_labels,
            unknown_labels=unknown_labels,
            dataset_name=self.dataset_name,
            num_known_classes=self.num_known_classes,
            model_path=str(self.model_path),
            timestamp=timestamp,
            feature_dim=all_features.shape[1],
            n_known_samples=known_features.shape[0],
            n_unknown_samples=unknown_features.shape[0]
        )

        print(f"\nSecond-to-last layer features saved to: {features_filepath}")
        print(f"   - Feature dimension: {all_features.shape[1]}")
        print(f"   - Known samples: {known_features.shape[0]}")
        print(f"   - Unknown samples: {unknown_features.shape[0]}")
        print(f"   - Total samples: {all_features.shape[0]}")

        return features_filepath

    def run_ood_detection(self):
        """Run out-of-distribution detection"""
        print("Starting OOD detection...")

        # Load model and data
        self.load_model_and_data()

        # Prepare data loaders
        data_loaders = self.prepare_data_loaders()

        # Check if unknown data exists
        if data_loaders['unknown'] is None:
            print("No unknown data available for OOD detection!")
            return None

        # Extract features
        print("Extracting features from training set...")
        train_features, train_labels, _ = self.extract_features_reference_style(data_loaders['train'])

        print("Extracting features from test set...")
        test_features, _, _ = self.extract_features_reference_style(data_loaders['test'])

        print("Extracting features from unknown set...")
        unknown_features, _, _ = self.extract_features_reference_style(data_loaders['unknown'])

        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")
        print(f"Unknown features shape: {unknown_features.shape}")

        # Use model config parameters
        d_token = self.model_config.get('d_token', 192)
        n_layers = self.model_config.get('n_layers', 5)
        last_layer_start = (n_layers - 1) * d_token
        last_layer_end = n_layers * d_token

        print(f"Using features from dimension {last_layer_start} to {last_layer_end} (last layer)")
        print(f"Model config: d_token={d_token}, n_layers={n_layers}")

        # Extract last layer features
        train_feat_last = train_features[:, last_layer_start:last_layer_end]
        test_feat_last = test_features[:, last_layer_start:last_layer_end]
        unknown_feat_last = unknown_features[:, last_layer_start:last_layer_end]

        # Compute scores
        begin = time.time()

        print("Computing scores for training set...")
        train_scores = self.compute_anomaly_scores(train_feat_last, train_labels, train_feat_last)

        print("Computing scores for test set (ID)...")
        known_scores = self.compute_anomaly_scores(train_feat_last, train_labels, test_feat_last)

        print("Computing scores for unknown set (OOD)...")
        unknown_scores = self.compute_anomaly_scores(train_feat_last, train_labels, unknown_feat_last)

        print(f"Score computation time: {time.time() - begin:.2f}s")

        print(f"Train scores - Mean: {train_scores.mean():.4f}, Std: {train_scores.std():.4f}")
        print(f"ID scores - Mean: {known_scores.mean():.4f}, Std: {known_scores.std():.4f}")
        print(f"OOD scores - Mean: {unknown_scores.mean():.4f}, Std: {unknown_scores.std():.4f}")

        # Convert to anomaly scores (higher is more anomalous)
        train_anomaly_scores = -train_scores  # training sample anomaly scores
        known_anomaly_scores = -known_scores  # ID sample anomaly scores
        unknown_anomaly_scores = -unknown_scores  # OOD sample anomaly scores

        # Save detection scores to NPZ file
        self.save_detection_scores(known_anomaly_scores, unknown_anomaly_scores)

        # Extract and save model features
        self.save_extracted_features()

        # Compute basic OOD detection metrics
        basic_results = metrics.cal_metric(known_anomaly_scores, unknown_anomaly_scores)

        print("\nOOD Detection Results:")
        print("=" * 60)
        print(f"FPR95: {basic_results['FPR95']:.4f}")
        print(f"AUROC: {basic_results['AUROC']:.4f}")
        print(f"AUPR:  {basic_results['AUPR']:.4f}")

        # Compute threshold-based classification metrics
        threshold_results = self.compute_threshold_based_metrics(train_anomaly_scores, known_anomaly_scores, unknown_anomaly_scores)

        # Merge all results
        results = {
            **basic_results,
            **threshold_results,
            'dataset_name': self.dataset_name,
            'num_known_classes': self.num_known_classes,
            'model_path': self.model_path
        }

        return results


def find_model_files(dataset_name, num_known_classes=None):
    """Find model files and configuration files"""
    model_files = []

    # New file organization structure
    if num_known_classes:
        model_dir = f"models/{dataset_name}/{num_known_classes}classes"
        if os.path.exists(model_dir):
            model_path = os.path.join(model_dir, f"frm_eta_{dataset_name}_ood_{num_known_classes}classes.pth")
            config_path = os.path.join(model_dir, f"config_{dataset_name}_ood_{num_known_classes}classes.json")
            if os.path.exists(model_path):
                model_files.append((model_path, config_path, num_known_classes))
    else:
        # Search all possible class numbers
        models_dir = f"models/{dataset_name}"
        if os.path.exists(models_dir):
            for class_dir in os.listdir(models_dir):
                if class_dir.endswith('classes'):
                    num_classes = int(class_dir.replace('classes', ''))
                    model_path = os.path.join(models_dir, class_dir, f"frm_eta_{dataset_name}_ood_{num_classes}classes.pth")
                    config_path = os.path.join(models_dir, class_dir, f"config_{dataset_name}_ood_{num_classes}classes.json")
                    if os.path.exists(model_path):
                        model_files.append((model_path, config_path, num_classes))

    # Old file organization structure (compatibility)
    old_model_path = f'models/frm_eta_{dataset_name}.pth'
    if os.path.exists(old_model_path) and not model_files:
        model_files.append((old_model_path, None, None))

    return model_files

def main():
    parser = argparse.ArgumentParser(description='FRM-ETA OOD Detection')
    parser.add_argument('--dataset', type=str, default='dapp',
                       choices=['dapp', 'mal'],
                       help='Dataset name')
    parser.add_argument('--known-classes', type=int, default=40,
                       help='Number of known classes') # Options: dapp:[1-44], mal:[1-23]
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model file')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--all', action='store_true',
                       help='Run OOD detection on all available models for the dataset')

    args = parser.parse_args()

    if args.all:
        # Run all available models
        model_files = find_model_files(args.dataset)
        if not model_files:
            print(f"No models found for dataset {args.dataset}")
            return

        print(f"Found {len(model_files)} models for dataset {args.dataset}")
        all_results = []

        for model_path, config_path, num_classes in model_files:
            print(f"\n{'='*80}")
            print(f"Running OOD detection with {num_classes} known classes")
            print(f"Model: {model_path}")
            print(f"{'='*80}")

            detector = FRMETAOODDetector(
                model_path=model_path,
                dataset_name=args.dataset,
                num_known_classes=num_classes,
                config_path=config_path
            )

            results = detector.run_ood_detection()
            if results:
                all_results.append(results)

        # Summarize all results
        if all_results:
            print(f"\n{'='*80}")
            print("SUMMARY OF ALL RESULTS")
            print(f"{'='*80}")
            print(f"{'Classes':<8} {'AUROC':<8} {'FPR95':<8} {'AUPR':<8} {'Accuracy':<10}")
            print("-" * 50)
            for result in all_results:
                print(f"{result['num_known_classes']:<8} {result['AUROC']:<8.4f} {result['FPR95']:<8.4f} {result['AUPR']:<8.4f} {result['accuracy']:<10.4f}")

    else:
        # Run single model
        if args.model_path:
            model_path = args.model_path
            config_path = args.config_path
        else:
            model_files = find_model_files(args.dataset, args.known_classes)
            if not model_files:
                print(f"No models found for dataset {args.dataset}")
                if args.known_classes:
                    print(f"with {args.known_classes} known classes")
                print("Please run training.py first to train the model.")
                return

            model_path, config_path, num_classes = model_files[0]
            args.known_classes = num_classes

        print(f"Using model: {model_path}")
        if config_path:
            print(f"Using config: {config_path}")

        # Create OOD detector
        detector = FRMETAOODDetector(
            model_path=model_path,
            dataset_name=args.dataset,
            num_known_classes=args.known_classes,
            config_path=config_path
        )

        # Run OOD detection
        results = detector.run_ood_detection()

        return results


if __name__ == "__main__":
    main()
