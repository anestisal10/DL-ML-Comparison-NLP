import torch
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def ensemble_predict(models, loader, device):
    """
    models: List of loaded PyTorch models
    loader: DataLoader
    device: torch.device
    """
    predictions = []
    for model in models:
        preds = get_predictions(model, loader, device)
        predictions.append(preds)
    
    # Stack predictions: [num_models, num_samples]
    predictions = np.vstack(predictions)
    
    # Hard Voting (Mode)
    # mode returns (mode_val, count)
    ensemble_preds, _ = mode(predictions, axis=0)
    ensemble_preds = ensemble_preds.ravel()
    
    return ensemble_preds

def evaluate_ensemble(models, loader, device, labels):
    preds = ensemble_predict(models, loader, device)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return acc, f1
