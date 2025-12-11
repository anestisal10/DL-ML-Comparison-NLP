# Sentiment Analysis Reproduction

This project reproduces the paper "A deep learning approach in predicting products’ sentiment ratings" using PyTorch.

## Setup

1.  **Virtual Environment (Windows)**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Preparation**:
    - Place `Womens Clothing E-Commerce Reviews.csv` in `data/raw/`.
    - Run preprocessing to create cleaned and augmented datasets:
    ```bash
    python -m src.preprocessing
    ```

4.  **Embeddings**:
    - The project uses `gensim.downloader` to automatically fetch `word2vec-google-news-300` and `fasttext-wiki-news-subwords-300`.
    - **Note**: The first run will download approx. 1.5GB of data. Ensure you have an internet connection.

## Running Experiments

Use `main.py` to run different configurations from the experimental matrix saving the results in `results/` as a csv file.

### Arguments
- `--mode`: `dl` (Deep Learning), `ml` (Machine Learning Baselines), `ensemble` (Ensemble of CNN+RNN+BiLSTM).
- `--model`: `cnn`, `rnn`, `bilstm`, `bert`, `roberta`, `albert`.
- `--dataset`: `original`, `augmented`.
- `--classes`: `3`, `5`.
- `--embedding`: `word2vec`, `fasttext`.

### Examples

**1. Standard Deep Learning (CNN, Word2Vec, Original, 3-Class)**
```bash
python main.py --mode dl --model cnn --dataset original --classes 3 --embedding word2vec
```

**2. Transformer (BERT, Original, 5-Class)**
```bash
python main.py --mode dl --model bert --dataset original --classes 5
```

**3. Ensemble (CNN + RNN + BiLSTM)**
*Note: This runs the "Best Setup" (Augmented, 3-Class, Word2Vec) as per the paper.*
```bash
python main.py --mode ensemble --dataset augmented --classes 3
```

**4. Machine Learning Baselines**
```bash
python main.py --mode ml --dataset augmented --classes 3
```
All the possible configurations are listed in `experiments.md`.

## Project Structure
- `src/preprocessing.py`: Data cleaning and EDA augmentation.
- `src/dataset.py`: PyTorch Dataset and embedding loading.
- `src/models.py`: Model architectures.
- `src/train.py`: Training loop and Cross-Validation.
- `src/ensemble.py`: Ensemble logic.
- `src/baselines.py`: Scikit-learn baselines.
