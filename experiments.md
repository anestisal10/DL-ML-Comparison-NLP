# Experimental Configurations

This document lists all possible command-line configurations for reproducing the experiments.

## 1. Standard Deep Learning Models
**Models**: CNN, RNN, Bi-LSTM
**Embeddings**: Word2Vec, FastText
**Datasets**: Original, Augmented
**Classes**: 3, 5

### CNN
```bash
# Word2Vec
python main.py --mode dl --model cnn --embedding word2vec --dataset original --classes 3
python main.py --mode dl --model cnn --embedding word2vec --dataset original --classes 5
python main.py --mode dl --model cnn --embedding word2vec --dataset augmented --classes 3
python main.py --mode dl --model cnn --embedding word2vec --dataset augmented --classes 5

# FastText
python main.py --mode dl --model cnn --embedding fasttext --dataset original --classes 3
python main.py --mode dl --model cnn --embedding fasttext --dataset original --classes 5
python main.py --mode dl --model cnn --embedding fasttext --dataset augmented --classes 3
python main.py --mode dl --model cnn --embedding fasttext --dataset augmented --classes 5
```

### RNN (LSTM)
```bash
# Word2Vec
python main.py --mode dl --model rnn --embedding word2vec --dataset original --classes 3
python main.py --mode dl --model rnn --embedding word2vec --dataset original --classes 5
python main.py --mode dl --model rnn --embedding word2vec --dataset augmented --classes 3
python main.py --mode dl --model rnn --embedding word2vec --dataset augmented --classes 5

# FastText
python main.py --mode dl --model rnn --embedding fasttext --dataset original --classes 3
python main.py --mode dl --model rnn --embedding fasttext --dataset original --classes 5
python main.py --mode dl --model rnn --embedding fasttext --dataset augmented --classes 3
python main.py --mode dl --model rnn --embedding fasttext --dataset augmented --classes 5
```

### Bi-LSTM
```bash
# Word2Vec
python main.py --mode dl --model bilstm --embedding word2vec --dataset original --classes 3
python main.py --mode dl --model bilstm --embedding word2vec --dataset original --classes 5
python main.py --mode dl --model bilstm --embedding word2vec --dataset augmented --classes 3
python main.py --mode dl --model bilstm --embedding word2vec --dataset augmented --classes 5

# FastText
python main.py --mode dl --model bilstm --embedding fasttext --dataset original --classes 3
python main.py --mode dl --model bilstm --embedding fasttext --dataset original --classes 5
python main.py --mode dl --model bilstm --embedding fasttext --dataset augmented --classes 3
python main.py --mode dl --model bilstm --embedding fasttext --dataset augmented --classes 5
```

## 2. Transformer Models
**Models**: BERT, RoBERTa, ALBERT
**Embeddings**: Internal (Fine-Tuned) - `--embedding` argument is ignored
**Datasets**: Original, Augmented
**Classes**: 3, 5

### BERT
```bash
python main.py --mode dl --model bert --dataset original --classes 3
python main.py --mode dl --model bert --dataset original --classes 5
python main.py --mode dl --model bert --dataset augmented --classes 3
python main.py --mode dl --model bert --dataset augmented --classes 5
```

### RoBERTa
```bash
python main.py --mode dl --model roberta --dataset original --classes 3
python main.py --mode dl --model roberta --dataset original --classes 5
python main.py --mode dl --model roberta --dataset augmented --classes 3
python main.py --mode dl --model roberta --dataset augmented --classes 5
```

### ALBERT
```bash
python main.py --mode dl --model albert --dataset original --classes 3
python main.py --mode dl --model albert --dataset original --classes 5
python main.py --mode dl --model albert --dataset augmented --classes 3
python main.py --mode dl --model albert --dataset augmented --classes 5
```

## 3. Ensemble Modeling
**Models**: Ensemble of CNN + RNN + Bi-LSTM
**Embeddings**: Word2Vec (Fixed)
**Datasets**: Original, Augmented
**Classes**: 3, 5

*Note: The paper identifies "Augmented / 3-Class" as the best setup.*

```bash
python main.py --mode ensemble --dataset original --classes 3
python main.py --mode ensemble --dataset original --classes 5
python main.py --mode ensemble --dataset augmented --classes 3
python main.py --mode ensemble --dataset augmented --classes 5
```

## 4. Machine Learning Baselines
**Models**: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM (All run sequentially)
**Embeddings**: Document Vectors (Averaged Word2Vec)
**Datasets**: Original, Augmented
**Classes**: 3, 5

```bash
python main.py --mode ml --dataset original --classes 3
python main.py --mode ml --dataset original --classes 5
python main.py --mode ml --dataset augmented --classes 3
python main.py --mode ml --dataset augmented --classes 5
```
