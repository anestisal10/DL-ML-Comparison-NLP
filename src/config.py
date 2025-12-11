import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')

# File Paths
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'Womens Clothing E-Commerce Reviews.csv')
CLEANED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'cleaned_data.csv')
AUGMENTED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'augmented_data.csv')

# Embedding Paths (Managed by gensim.downloader)
# WORD2VEC_PATH and FASTTEXT_PATH are no longer needed as we use api.load()

# Hyperparameters
MAX_SEQ_LEN = 100 # Adjust based on data analysis
EMBEDDING_DIM = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 3 # Early stopping patience
K_FOLDS = 10

# Random Seed
SEED = 42
