import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import gensim
from gensim.models import KeyedVectors
from src import config
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, tokenizer=None, max_len=config.MAX_SEQ_LEN):
        """
        Args:
            texts: List of strings
            labels: List of integers
            vocab: Dict mapping word -> index (for static embeddings)
            tokenizer: HuggingFace tokenizer (for transformers)
            max_len: Max sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.tokenizer:
            # Transformer Tokenization
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        
        elif self.vocab:
            # Static Embedding Tokenization
            tokens = text.split()
            # Pad or truncate
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            
            # Convert to indices
            indices = [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens]
            
            # Padding
            if len(indices) < self.max_len:
                indices += [self.vocab.get('<PAD>', 0)] * (self.max_len - len(indices))
                
            return {
                'input_ids': torch.tensor(indices, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            raise ValueError("Either vocab or tokenizer must be provided.")

def build_vocab(texts, min_freq=1):
    """Builds vocabulary from a list of texts."""
    word_counts = {}
    for text in texts:
        for word in str(text).split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocab mapping
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in sorted_words:
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
            
    return vocab

import gensim.downloader as api

def load_embedding_matrix(vocab, embedding_type='word2vec'):
    """
    Loads pre-trained embeddings using gensim.downloader and creates a matrix matching the vocab.
    """
    print(f"Loading {embedding_type} embeddings via gensim.downloader...")
    
    if embedding_type == 'word2vec':
        model_name = 'word2vec-google-news-300'
    elif embedding_type == 'fasttext':
        model_name = 'fasttext-wiki-news-subwords-300'
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
        
    try:
        print(f"Fetching/Loading {model_name} (this may take a while first time)...")
        word_vectors = api.load(model_name)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return np.random.uniform(-0.25, 0.25, (len(vocab), config.EMBEDDING_DIM))

    embedding_matrix = np.zeros((len(vocab), config.EMBEDDING_DIM))
    hits = 0
    misses = 0
    
    for word, i in vocab.items():
        if i < 2: continue # Skip PAD and UNK
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
            hits += 1
        except KeyError:
            # Initialize random for unknown
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, config.EMBEDDING_DIM)
            misses += 1
            
    print(f"Embedding loaded. Hits: {hits}, Misses: {misses}")
    return embedding_matrix
