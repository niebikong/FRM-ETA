import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

from bin.frm_eta_ood import FRMETAOOD
from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer


def seed_everything(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_and_preprocess_data(dataset_name, cache_path=None):
    """Load and preprocess dataset with caching support"""
    if cache_path:
        cache_file = cache_path + '.npz'
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            cached_data = np.load(cache_file)
            return cached_data['X'], cached_data['y']

    print(f"Processing raw data for dataset: {dataset_name}")

    dataset_paths = {
        'dapp': r'data/dapp_84_features_label.csv',
        'mal': r'data/label_encodered_malicious_TLS-1.csv',
    }

    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_paths.keys())}")

    output_path = dataset_paths[dataset_name]

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Dataset file not found: {output_path}")

    print(f"Loading data from: {output_path}")
    data = pd.read_csv(output_path, dtype=str)
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col].astype(str))

    data = data.astype(np.float64)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(np.int64)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, X=X_scaled, y=y)
        print(f"Cached processed data to {cache_path}")

    return X_scaled, y


def prepare_ood_dataset(dataset_name, num_known_classes=5, use_cache=True):
    """Prepare dataset for out-of-distribution detection"""
    print(f"Loading {dataset_name} dataset for OOD detection v2...")

    # Use cache to speed up data loading, file name contains dataset name
    cache_path = f'data/cache/{dataset_name}_processed' if use_cache else None
    X, y = load_and_preprocess_data(dataset_name, cache_path)

    # Get class information
    unique_classes = np.unique(y)
    print(f"Total classes: {len(unique_classes)}")
    print(f"Classes: {unique_classes}")

    # Check if number of known classes is reasonable
    if num_known_classes > len(unique_classes):
        raise ValueError(f"num_known_classes ({num_known_classes}) must be less than total classes ({len(unique_classes)})")

    # Specify number of classes as known classes, remaining classes as unknown classes
    known_classes = unique_classes[:num_known_classes]
    unknown_classes = unique_classes[num_known_classes:]

    print(f"Known classes: {len(known_classes)} classes - {known_classes}")
    print(f"Unknown classes: {len(unknown_classes)} classes - {unknown_classes}")

    # Use vectorized operations to improve efficiency
    known_mask = np.isin(y, known_classes)
    unknown_mask = np.isin(y, unknown_classes)

    X_known = X[known_mask]
    y_known = y[known_mask]
    X_unknown = X[unknown_mask]
    y_unknown = y[unknown_mask]

    print(f"Known samples: {len(X_known)}")
    print(f"Unknown samples: {len(X_unknown)}")

    # Use vectorized operations to remap labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}
    y_known_mapped = np.vectorize(label_mapping.get)(y_known)

    # Split known class data into train/validation/test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_known, y_known_mapped, test_size=0.3, random_state=42, stratify=y_known_mapped
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Create data directory, named using dataset name and number of known classes
    dataset_dir_name = f"{dataset_name}_ood_{num_known_classes}classes"
    data_dir = Path('data') / dataset_dir_name
    data_dir.mkdir(exist_ok=True)

    # Save known class data
    np.save(data_dir / 'X_num_train.npy', X_train)
    np.save(data_dir / 'X_num_val.npy', X_val)
    np.save(data_dir / 'X_num_test.npy', X_test)

    np.save(data_dir / 'y_train.npy', y_train.astype(np.int64))
    np.save(data_dir / 'y_val.npy', y_val.astype(np.int64))
    np.save(data_dir / 'y_test.npy', y_test.astype(np.int64))

    # Save unknown class data (for OOD detection)
    np.save(data_dir / 'X_unknown.npy', X_unknown)
    np.save(data_dir / 'y_unknown.npy', y_unknown.astype(np.int64))

    # Create index files
    np.save(data_dir / 'idx_train.npy', np.arange(len(X_train)))
    np.save(data_dir / 'idx_val.npy', np.arange(len(X_val)))
    np.save(data_dir / 'idx_test.npy', np.arange(len(X_test)))

    # Create info.json
    info = {
        "name": dataset_dir_name,
        "id": f"{dataset_dir_name}--default",
        "task_type": "multiclass",
        "n_num_features": X_train.shape[1],
        "n_cat_features": 0,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "n_classes": num_known_classes,
        "unknown_samples": len(X_unknown),
        "original_dataset": dataset_name,
        "total_classes": len(unique_classes)
    }

    with open(data_dir / 'info.json', 'w') as f:
        json.dump(info, f, indent=4)

    # Create READY file
    (data_dir / 'READY').touch()

    print(f"OOD dataset created successfully!")
    print(f"Dataset directory: {data_dir}")
    print(f"Train size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    print(f"Unknown size: {len(X_unknown)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Known classes: {num_known_classes}")

    return info, dataset_dir_name


def save_evaluation_results(true_labels, predictions, model_dir, dataset_dir_name, test_accuracy, num_known_classes):
    """Save evaluation results including confusion matrix and metrics"""

    cm = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, output_dict=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cm_filename = f"confusion_matrix_training_{dataset_dir_name}_{timestamp}.txt"
    cm_filepath = model_dir / cm_filename

    # Prepare content to save
    content = []
    content.append("=" * 80)
    content.append("TRAINING PHASE CONFUSION MATRIX AND CLASSIFICATION REPORT")
    content.append("=" * 80)
    content.append(f"Dataset: {dataset_dir_name}")
    content.append(f"Known Classes: {num_known_classes}")
    content.append(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    content.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append("")

    content.append("CONFUSION MATRIX:")
    content.append("Rows: True labels, Columns: Predicted labels")
    content.append("")

    # Add column headers
    header = "     " + "".join([f"{i:6d}" for i in range(num_known_classes)])
    content.append(header)

    # Add confusion matrix rows
    for i in range(num_known_classes):
        row = f"{i:3d}: " + "".join([f"{cm[i,j]:6d}" for j in range(num_known_classes)])
        content.append(row)

    content.append("")
    content.append("CLASSIFICATION METRICS:")
    content.append("")

    # Add metrics for each class
    for class_id in range(num_known_classes):
        if str(class_id) in class_report:
            metrics = class_report[str(class_id)]
            content.append(f"Class {class_id}:")
            content.append(f"  Precision: {metrics['precision']:.4f}")
            content.append(f"  Recall:    {metrics['recall']:.4f}")
            content.append(f"  F1-Score:  {metrics['f1-score']:.4f}")
            content.append(f"  Support:   {metrics['support']}")
            content.append("")

    # Add overall metrics
    if 'macro avg' in class_report:
        macro_avg = class_report['macro avg']
        content.append("MACRO AVERAGE:")
        content.append(f"  Precision: {macro_avg['precision']:.4f}")
        content.append(f"  Recall:    {macro_avg['recall']:.4f}")
        content.append(f"  F1-Score:  {macro_avg['f1-score']:.4f}")
        content.append("")

    if 'weighted avg' in class_report:
        weighted_avg = class_report['weighted avg']
        content.append("WEIGHTED AVERAGE:")
        content.append(f"  Precision: {weighted_avg['precision']:.4f}")
        content.append(f"  Recall:    {weighted_avg['recall']:.4f}")
        content.append(f"  F1-Score:  {weighted_avg['f1-score']:.4f}")
        content.append("")

    content.append("=" * 80)

    with open(cm_filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

    print(f"\nTraining confusion matrix saved to: {cm_filepath}")
    return cm_filepath


def train_model(dataset_dir_name, original_dataset_name, num_known_classes, use_mixed_precision=True, accumulation_steps=1):
    """Train FRM-ETA model for out-of-distribution detection"""
    # Setup parameters
    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()  
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    seed_everything(42)

    # Load dataset with caching enabled for efficiency
    normalization = 'quantile'
    transformation = Transformations(normalization=normalization)
    dataset = build_dataset(DATA / dataset_dir_name, transformation, True)  # Enable caching

    print(f"Dataset loaded: {dataset.n_features} features, {dataset.n_classes} classes")
    print(f"Train: {dataset.size('train')}, Val: {dataset.size('val')}, Test: {dataset.size('test')}")

    # Handle data type compatibility
    print(f"Original data type: {dataset.X_num['train'].dtype}")

    # If using mixed precision training, ensure data type compatibility
    if use_mixed_precision and dataset.X_num['train'].dtype == np.float64:
        print("Converting data to float32 for mixed precision compatibility...")
        dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
        print(f"Data type after conversion: {dataset.X_num['train'].dtype}")
    else:
        print("Keeping original data precision")

    d_out = dataset.n_classes or 1
    X_num, X_cat, ys = prepare_tensors(dataset, device=device)


    batch_size = 256  
    val_batch_size = 512  

    # Create data loaders,
    data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
    train_dataset = TensorDataset(*(d['train'] for d in data_list))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,  
        num_workers=0  
    )

    val_dataset = TensorDataset(*(d['val'] for d in data_list))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )

    test_dataset = TensorDataset(*(d['test'] for d in data_list))
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    # Create model
    n_num_features = dataset.n_num_features
    cardinalities = dataset.get_category_sizes('train')
    n_categories = len(cardinalities)
    cardinalities = None if n_categories == 0 else cardinalities

    # Model configuration - based on best configuration from hyperparameter tuning 
    model_config = {
        'd_numerical': n_num_features,
        'categories': cardinalities,
        'd_out': d_out,
        'activation': 'reglu',
        'initialization': 'kaiming',
        'n_heads': 12,  # Best config: 12 heads
        'prenormalization': True,
        'residual_dropout': 0.1,  # Best config: 0.1 
        'attention_dropout': 0.4,  # Keep 0.4
        'd_ffn_factor': 2.0,  # Best config: 2.0
        'd_token': 192,  # Best config: 192 token dim
        'ffn_dropout': 0.1,  # Keep 0.1
        'n_layers': 5,  # Best config: 5 layers
        'token_bias': True,
        'kv_compression': None,
        'kv_compression_sharing': None
    }

    model = FRMETAOOD(**model_config).to(device)

    # Test feature_list method
    print("Testing feature_list method...")
    dummy_input = torch.randn(2, n_num_features).to(device)
    logits, feature_list = model.feature_list(dummy_input, None)
    print(f"Logits shape: {logits.shape}")
    # print(f"Feature list length: {len(feature_list)}")
    # for i, feat in enumerate(feature_list):
    #     print(f"Layer {i} feature shape: {feat.shape}")
    
    # Optimizer
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    def needs_small_lr(name):
        return any(x in name for x in ['.col_head', '.col_tail'])

    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k) and not needs_small_lr(k)]
    parameters_with_slr = [v for k, v in model.named_parameters() if needs_small_lr(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]

    # Optimizer configuration - based on best configuration from hyperparameter tuning
    optimizer = make_optimizer(
        'adamw',
        [
            {'params': parameters_with_wd},
            {'params': parameters_with_slr, 'lr': 0.01, 'weight_decay': 0.0},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ],
        5e-4,  # Best config: 5e-4 learning rate
        1e-4,  # Best config: 1e-4 weight decay
    )

    # Loss function
    loss_fn = F.cross_entropy

    # Initialize mixed precision training
    scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None

    # Training parameters
    n_epochs = 30
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    print("Starting optimized training...")
    print(f"Mixed precision: {use_mixed_precision and torch.cuda.is_available()}")
    print(f"Gradient accumulation steps: {accumulation_steps}")

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            x_num, x_cat, y = (batch[0], None, batch[1]) if len(batch) == 2 else batch

            # Use mixed precision training
            if scaler is not None:
                with autocast():
                    outputs = model(x_num, x_cat)
                    loss = loss_fn(outputs, y) / accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(x_num, x_cat)
                loss = loss_fn(outputs, y) / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                x_num, x_cat, y = (batch[0], None, batch[1]) if len(batch) == 2 else batch

                if scaler is not None:
                    with autocast():
                        outputs = model(x_num, x_cat)
                        loss = loss_fn(outputs, y)
                else:
                    outputs = model(x_num, x_cat)
                    loss = loss_fn(outputs, y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start_time

        print(f'Epoch [{epoch+1}/{n_epochs}] '
              f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, '
              f'Time: {epoch_time:.2f}s')

        # Early stopping and best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")

    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0

    # Collect all predictions and true labels for confusion matrix
    all_predictions = []
    all_true_labels = []

    print("Running final test evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            x_num, x_cat, y = (batch[0], None, batch[1]) if len(batch) == 2 else batch

            if scaler is not None:
                with autocast():
                    outputs = model(x_num, x_cat)
                    loss = loss_fn(outputs, y)
            else:
                outputs = model(x_num, x_cat)
                loss = loss_fn(outputs, y)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()

            # Collect predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())

    test_acc = test_correct / test_total
    test_loss = test_loss / len(test_loader)

    model_dir = Path('models') / original_dataset_name / f'{num_known_classes}classes'
    model_dir.mkdir(parents=True, exist_ok=True)

    save_evaluation_results(all_true_labels, all_predictions, model_dir, dataset_dir_name, test_acc, num_known_classes)

    # Print training results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f'Final Test Accuracy: {test_acc:.4f}')
    print(f'Final Test Loss: {test_loss:.4f}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')

    # Save model
    model_path = model_dir / f'frm_eta_{dataset_dir_name}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # Save training configuration
    config_path = model_dir / f'config_{dataset_dir_name}.json'
    training_config = {
        'dataset_name': original_dataset_name,
        'dataset_dir_name': dataset_dir_name,
        'num_known_classes': num_known_classes,
        'model_config': {
            'd_numerical': dataset.n_num_features,
            'n_classes': dataset.n_classes,
            'n_features': dataset.n_features
        },
        'training_params': {
            'use_mixed_precision': use_mixed_precision,
            'accumulation_steps': accumulation_steps,
            'batch_size': batch_size,
            'val_batch_size': val_batch_size,
            'patience': patience
        }
    }
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)
    print(f'Training configuration saved to {config_path}')

    # Save training statistics
    stats_path = model_dir / f'stats_{dataset_dir_name}.json'
    training_stats = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,

        'dataset_info': {
            'original_dataset': original_dataset_name,
            'dataset_dir': dataset_dir_name,
            'num_known_classes': num_known_classes,
            'total_features': dataset.n_features
        }
    }
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=4)
    print(f'Training statistics saved to {stats_path}')

    return model, dataset, dataloaders


def main():
    """Main training function, supports multiple datasets and configurable number of known classes"""
    # Configuration parameters
    config = {
        'dataset_name': 'dapp',  # Options: dapp:[1-44], mal:[1-23]
        'num_known_classes': 40,  # Number of known classes
        'use_mixed_precision': True,
        'accumulation_steps': 1,
        'use_cache': True
    }

    # Dataset information
    dataset_info = {
        'dapp': 'data/dapp_84_features_label.csv',
        'mal': 'data/label_encodered_malicious_TLS-1.csv',
    }

    dataset_name = config['dataset_name']
    num_known_classes = config['num_known_classes']

    print("=" * 60)
    print("FRM-ETA OOD Training Pipeline")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Data file: {dataset_info.get(dataset_name, 'Unknown')}")
    print(f"Known classes: {num_known_classes}")
    print(f"Mixed precision: {config['use_mixed_precision']}")
    print(f"Cache enabled: {config['use_cache']}")
    print("=" * 60)

    try:
        # Create OOD dataset
        print("\n1. Creating OOD dataset...")
        _, dataset_dir_name = prepare_ood_dataset(
            dataset_name,
            num_known_classes=num_known_classes,
            use_cache=config['use_cache']
        )

        # Train model
        print("\n2. Starting model training...")
        train_model(
            dataset_dir_name,
            dataset_name,
            num_known_classes,
            use_mixed_precision=config['use_mixed_precision'],
            accumulation_steps=config['accumulation_steps']
        )

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model saved in: models/{dataset_name}/{num_known_classes}classes/")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
