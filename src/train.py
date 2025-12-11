import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import copy
from tqdm import tqdm
from src import config

def train_one_epoch(model, loader, criterion, optimizer, device, is_transformer=False, l1_lambda=0.0):
    model.train()
    running_loss = 0.0
    
    for batch in loader:
        if is_transformer:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
        else:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # L1 Regularization for CNN (if l1_lambda > 0)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, is_transformer=False, num_classes=3):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            if is_transformer:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = running_loss / len(loader.dataset)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc = 0.0 # Handle cases where not all classes are present
        
    return avg_loss, acc, prec, rec, f1, auc

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, patience=3, is_transformer=False, l1_lambda=0.0, num_classes=3):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, is_transformer, l1_lambda)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion, device, is_transformer, num_classes)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
        
        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
                
    model.load_state_dict(best_model_wts)
    return model, history

def cross_validation(model_class, dataset, k_folds=config.K_FOLDS, batch_size=32, num_classes=3, embedding_matrix=None, model_params={}, is_transformer=False):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.SEED)
    
    results = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # For StratifiedKFold, we need X and y. X can be indices.
    indices = np.arange(len(dataset))
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"Fold {fold+1}/{k_folds}")
        
        # Subset datasets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        
        # Initialize model
        if is_transformer:
            model = model_class(**model_params)
        else:
            model = model_class(embedding_matrix, num_classes, **model_params)
            
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        
        # L1 lambda for CNN
        l1_lambda = 0.0
        if 'CNN' in model.__class__.__name__:
             # Paper specifies L1 regularization. Value not specified, assuming small.
             l1_lambda = 1e-5
        
        trained_model, _ = train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                                       epochs=config.EPOCHS, patience=config.PATIENCE, 
                                       is_transformer=is_transformer, l1_lambda=l1_lambda, num_classes=num_classes)
        
        # Final Evaluation
        _, acc, prec, rec, f1, auc = evaluate(trained_model, val_loader, criterion, device, is_transformer, num_classes)
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
        results['auc'].append(auc)
        
        print(f"Fold {fold+1} Results - Acc: {acc:.4f}, F1: {f1:.4f}")
        
    # Average results
    avg_results = {k: np.mean(v) for k, v in results.items()}
    print("\nAverage Cross-Validation Results:")
    for k, v in avg_results.items():
        print(f"{k}: {v:.4f}")
        
    return avg_results
