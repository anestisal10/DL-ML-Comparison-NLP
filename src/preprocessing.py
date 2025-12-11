import pandas as pd
import numpy as np
import nltk
import re
import os
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from src import config

# Ensure NLTK resources are available
def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res)

download_nltk_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize (split by space)
    words = text.split()
    
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# --- EDA Implementation ---

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    return sentence

def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return ' '.join(new_words)

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return ' '.join(new_words)

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return ' '.join(new_words)

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

def eda_augment(sentence, num_aug=4, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
    words = sentence.split()
    num_words = len(words)
    if num_words == 0:
        return [sentence] * num_aug

    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    augmented_sentences = []
    
    # SR
    for _ in range(num_aug // 4):
        augmented_sentences.append(synonym_replacement(words, n_sr))
    # RI
    for _ in range(num_aug // 4):
        augmented_sentences.append(random_insertion(words, n_ri))
    # RS
    for _ in range(num_aug // 4):
        augmented_sentences.append(random_swap(words, n_rs))
    # RD
    for _ in range(num_aug // 4):
        augmented_sentences.append(random_deletion(words, p_rd))
        
    # Fill remaining if needed (due to integer division)
    while len(augmented_sentences) < num_aug:
        augmented_sentences.append(random_swap(words, n_rs)) # Default to swap

    return augmented_sentences

# --- Label Mapping ---

def map_labels(rating):
    # 5-Class: 1-5 -> 0-4
    label_5 = int(rating) - 1
    
    # 3-Class: 1,2 -> 0 (Neg); 3 -> 1 (Neu); 4,5 -> 2 (Pos)
    if rating <= 2:
        label_3 = 0
    elif rating == 3:
        label_3 = 1
    else:
        label_3 = 2
        
    return label_5, label_3

def process_data():
    if not os.path.exists(config.RAW_DATA_FILE):
        print(f"Error: Raw data file not found at {config.RAW_DATA_FILE}")
        return

    print("Loading raw data...")
    df = pd.read_csv(config.RAW_DATA_FILE)
    
    # Assuming the review text column is 'Review Text' and rating is 'Rating'
    # Adjust column names if necessary based on actual CSV
    if 'Review Text' not in df.columns or 'Rating' not in df.columns:
        print("Error: Expected columns 'Review Text' and 'Rating' not found.")
        print(f"Available columns: {df.columns}")
        return

    # Drop missing values
    df = df.dropna(subset=['Review Text', 'Rating'])

    print("Cleaning text...")
    df['cleaned_text'] = df['Review Text'].apply(clean_text)
    
    # Label Mapping
    print("Mapping labels...")
    df[['label_5', 'label_3']] = df['Rating'].apply(lambda x: pd.Series(map_labels(x)))

    # Save Cleaned Data (Original)
    print(f"Saving cleaned data to {config.CLEANED_DATA_FILE}...")
    df.to_csv(config.CLEANED_DATA_FILE, index=False)

    # Augmentation
    print("Performing Data Augmentation (EDA)...")
    augmented_rows = []
    
    # Only augment training data? The plan says "run the EDA algorithm... on the training data offline".
    # However, we haven't split train/test yet. Usually, we augment ONLY training data.
    # But the prompt says "Result: You will have two distinct CSV/DataFrame objects: one Original and one Augmented".
    # And later "10-Fold Cross-Validation".
    # If we augment everything now, we risk data leakage if we split later.
    # BUT, for 10-Fold CV, we need to split *then* augment the training fold.
    # OR, the prompt implies creating a static Augmented Dataset.
    # "Result: You will have two distinct CSV/DataFrame objects: one Original and one Augmented (which is roughly 5x larger)."
    # If I augment everything now, I must ensure that during CV, I don't have augmented versions of validation samples in the training set.
    # This is tricky with pre-augmented files.
    # Standard practice for static augmented datasets in papers: Augment everything, but when splitting, ensure groups stay together.
    # OR, just augment everything and assume the user knows to handle it?
    # Actually, the prompt says "run the EDA algorithm... on the training data offline".
    # This implies we should probably split first?
    # But the "Experimental Matrix" lists "Dataset Version: Augmented".
    # If I create a single 'augmented_data.csv', I should probably include an ID to group original and augmented versions.
    
    # Let's add an ID column to track original samples.
    df['original_id'] = df.index
    
    for idx, row in df.iterrows():
        # Add original
        augmented_rows.append({
            'cleaned_text': row['cleaned_text'],
            'label_5': row['label_5'],
            'label_3': row['label_3'],
            'original_id': row['original_id'],
            'is_augmented': False
        })
        
        # Augment
        aug_sentences = eda_augment(row['cleaned_text'], num_aug=4)
        for aug_sent in aug_sentences:
            augmented_rows.append({
                'cleaned_text': aug_sent,
                'label_5': row['label_5'],
                'label_3': row['label_3'],
                'original_id': row['original_id'],
                'is_augmented': True
            })
            
    aug_df = pd.DataFrame(augmented_rows)
    print(f"Saving augmented data to {config.AUGMENTED_DATA_FILE}...")
    aug_df.to_csv(config.AUGMENTED_DATA_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    process_data()
