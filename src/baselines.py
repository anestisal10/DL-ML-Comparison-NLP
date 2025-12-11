import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src import config
from src.dataset import load_embedding_matrix, build_vocab

def get_document_vectors(texts, embedding_matrix, vocab):
    """
    Compute averaged word vectors for each document.
    """
    doc_vectors = []
    vocab_inv = {v: k for k, v in vocab.items()}
    
    for text in texts:
        words = str(text).split()
        vectors = []
        for word in words:
            if word in vocab:
                idx = vocab[word]
                if idx < 2: continue # Skip PAD/UNK if they are just placeholders
                vectors.append(embedding_matrix[idx])
        
        if vectors:
            doc_vectors.append(np.mean(vectors, axis=0))
        else:
            doc_vectors.append(np.zeros(config.EMBEDDING_DIM))
            
    return np.array(doc_vectors)

def run_baselines(texts, labels, embedding_matrix, vocab, k_folds=10):
    X = get_document_vectors(texts, embedding_matrix, vocab)
    y = np.array(labels)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True)
    }
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.SEED)
    
    results = {}
    
    for name, model in models.items():
        print(f"Running {name}...")
        fold_accs = []
        fold_f1s = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            fold_accs.append(accuracy_score(y_val, preds))
            fold_f1s.append(f1_score(y_val, preds, average='weighted'))
            
        results[name] = {
            'Accuracy': np.mean(fold_accs),
            'F1': np.mean(fold_f1s)
        }
        print(f"{name} - Acc: {np.mean(fold_accs):.4f}, F1: {np.mean(fold_f1s):.4f}")
        
    return results
