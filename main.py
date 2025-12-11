import argparse
import pandas as pd
import torch
import numpy as np
import os
import csv
import src.config as config
from src.preprocessing import process_data
from src.dataset import SentimentDataset, load_embedding_matrix, build_vocab
from src.models import SentimentCNN, SentimentRNN, SentimentBiLSTM, TransformerClassifier
from src.train import cross_validation, train_model, evaluate
from src.baselines import run_baselines
from src.ensemble import ensemble_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

def load_data(args):
    if args.dataset == 'original':
        path = config.CLEANED_DATA_FILE
    else:
        path = config.AUGMENTED_DATA_FILE
        
    if not os.path.exists(path):
        print(f"Data file {path} not found. Running preprocessing...")
        process_data()
        
    df = pd.read_csv(path)
    
    # Filter for class setup
    if args.classes == 3:
        label_col = 'label_3'
    else:
        label_col = 'label_5'
        
    # Determine text column
    if 'cleaned_text' in df.columns:
        text_col = 'cleaned_text'
    elif 'text' in df.columns:
        text_col = 'text'
    else:
        raise KeyError("Neither 'cleaned_text' nor 'text' column found in dataset.")
        
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()
    
    return texts, labels

def run_deep_learning(args, texts, labels):
    print(f"Running Deep Learning Experiment: {args.model} | {args.embedding} | {args.dataset} | {args.classes}-Class")
    
    # Prepare Features
    if args.model in ['bert', 'roberta', 'albert']:
        # Transformer
        from transformers import AutoTokenizer
        model_name_map = {
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'albert': 'albert-base-v2'
        }
        model_name = model_name_map[args.model]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = SentimentDataset(texts, labels, tokenizer=tokenizer)
        
        model_class = TransformerClassifier
        model_params = {'model_name': model_name, 'num_classes': args.classes}
        embedding_matrix = None
        is_transformer = True
        
    else:
        # Standard DL
        vocab = build_vocab(texts)
        embedding_matrix = load_embedding_matrix(vocab, args.embedding)
        dataset = SentimentDataset(texts, labels, vocab=vocab)
        
        model_map = {
            'cnn': SentimentCNN,
            'rnn': SentimentRNN,
            'bilstm': SentimentBiLSTM
        }
        model_class = model_map[args.model]
        model_params = {}
        is_transformer = False
        
    # Run CV
    results = cross_validation(model_class, dataset, num_classes=args.classes, 
                               embedding_matrix=embedding_matrix, model_params=model_params, 
                               is_transformer=is_transformer)
    return results

def run_ensemble_cv(texts, labels, args):
    print("Running Ensemble CV (CNN + RNN + BiLSTM)...")
    # Setup for Standard DL (Word2Vec / Augmented / 3-Class as per paper best setup)
    # Assuming args are already set to this or we force them
    vocab = build_vocab(texts)
    embedding_matrix = load_embedding_matrix(vocab, 'word2vec')
    dataset = SentimentDataset(texts, labels, vocab=vocab)
    
    skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
    
    fold_accs = []
    fold_f1s = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    indices = np.arange(len(dataset))
    y = np.array(labels)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y)):
        print(f"Ensemble Fold {fold+1}/10")
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=val_subsampler)
        
        # Train 3 models
        models = []
        for m_name, m_class in [('cnn', SentimentCNN), ('rnn', SentimentRNN), ('bilstm', SentimentBiLSTM)]:
            print(f"  Training {m_name}...")
            model = m_class(embedding_matrix, args.classes).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
            
            l1 = 1e-5 if m_name == 'cnn' else 0.0
            
            trained_model, _ = train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                                           epochs=config.EPOCHS, patience=config.PATIENCE, l1_lambda=l1, num_classes=args.classes)
            models.append(trained_model)
            
        # Ensemble Prediction
        print("  Evaluating Ensemble...")
        # Get labels for validation set (order matches val_loader)
        val_labels = []
        for batch in val_loader:
            val_labels.extend(batch['label'].tolist())
            
        preds = ensemble_predict(models, val_loader, device)
        
        acc = accuracy_score(val_labels, preds)
        f1 = f1_score(val_labels, preds, average='weighted')
        
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold+1} Ensemble - Acc: {acc:.4f}, F1: {f1:.4f}")
        
    avg_acc = np.mean(fold_accs)
    avg_f1 = np.mean(fold_f1s)
    print(f"\nEnsemble CV Results - Avg Acc: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")
    return {'accuracy': avg_acc, 'f1': avg_f1}

def save_results(args, metrics, filename=f'results/results.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Define headers
        headers = ['Mode', 'Model', 'Dataset', 'Classes', 'Embedding', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        if not file_exists:
            writer.writerow(headers)
        
        if args.mode == 'ml':
            # metrics is {model_name: {Accuracy, F1}}
            for model_name, scores in metrics.items():
                row = ['ml', model_name, args.dataset, args.classes, 'word2vec (doc)', scores['Accuracy'], '', '', scores['F1'], '']
                writer.writerow(row)
        else:
            # DL or Ensemble
            row = [
                args.mode,
                args.model if args.mode == 'dl' else 'ensemble',
                args.dataset,
                args.classes,
                args.embedding if args.mode == 'dl' else 'word2vec',
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0),
                metrics.get('auc', 0)
            ]
            writer.writerow(row)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Reproduction')
    
    parser.add_argument('--mode', type=str, default='dl', choices=['dl', 'ml', 'ensemble'], help='Experiment mode')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'rnn', 'bilstm', 'bert', 'roberta', 'albert'], help='Model architecture')
    parser.add_argument('--dataset', type=str, default='original', choices=['original', 'augmented'], help='Dataset version')
    parser.add_argument('--classes', type=int, default=3, choices=[3, 5], help='Number of classes')
    parser.add_argument('--embedding', type=str, default='word2vec', choices=['word2vec', 'fasttext'], help='Embedding type')
    
    args = parser.parse_args()
    
    # Check Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device Check ---")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    else:
        print("WARNING: Running on CPU. This will be slow.")
        print("If you have a GPU, ensure you have installed the CUDA version of PyTorch.")
    print(f"--------------------")
    
    texts, labels = load_data(args)
    
    if args.mode == 'dl':
        results = run_deep_learning(args, texts, labels)
        save_results(args, results, filename=f'results/{args.model}_{args.embedding}_{args.dataset}_{args.classes}.csv')
    elif args.mode == 'ml':
        print("Running ML Baselines...")
        # ML baselines use Document Vectors (averaged Word2Vec)
        vocab = build_vocab(texts)
        embedding_matrix = load_embedding_matrix(vocab, 'word2vec')
        results = run_baselines(texts, labels, embedding_matrix, vocab)
        save_results(args, results,filename=f'results/{args.model}_{args.embedding}_{args.dataset}_{args.classes}.csv')
    elif args.mode == 'ensemble':
        results = run_ensemble_cv(texts, labels, args)
        save_results(args, results, filename=f'results/{args.model}_{args.embedding}_{args.dataset}_{args.classes}.csv')

if __name__ == "__main__":
    main()
