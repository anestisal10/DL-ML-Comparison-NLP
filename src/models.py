import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig
from src import config

class SentimentCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, freeze_embeddings=True):
        super(SentimentCNN, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=freeze_embeddings)
        
        # Parallel Conv1d layers
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=5)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256 * 3, num_classes)

    def forward(self, x):
        # x: [Batch, Length]
        x = self.embedding(x) # [Batch, Length, Dim]
        x = x.permute(0, 2, 1) # [Batch, Dim, Length]
        
        # Convolution + ReLU + MaxPool
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2) # [Batch, 256]
        
        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        
        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        
        # Concatenate
        x_cat = torch.cat((x1, x2, x3), dim=1) # [Batch, 256*3]
        
        x_cat = self.dropout(x_cat)
        logits = self.fc(x_cat)
        return logits

class SentimentRNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, freeze_embeddings=True):
        super(SentimentRNN, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=freeze_embeddings)
        
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x) # [Batch, Length, Dim]
        _, (hidden, _) = self.lstm(x)
        # hidden: [num_layers * num_directions, Batch, Hidden] -> [1, Batch, 256]
        x = hidden[-1] # [Batch, 256]
        
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class SentimentBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, num_classes, freeze_embeddings=True):
        super(SentimentBiLSTM, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=freeze_embeddings)
        
        # Paper says 64 units for Bi-LSTM
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        
        self.dense = nn.Linear(64 * 2, 64) # Intermediate dense layer
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # output: [Batch, Length, Hidden*2]
        # hidden: [2, Batch, Hidden]
        output, (hidden, _) = self.lstm(x)
        
        # Concatenate final forward and backward hidden states
        # hidden[-2] is forward, hidden[-1] is backward
        x = torch.cat((hidden[-2], hidden[-1]), dim=1) # [Batch, 128]
        
        x = F.relu(self.dense(x)) # [Batch, 64]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TransformerClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
