import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # HTML
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # URLs
    text = re.sub(r'[^a-z\s]', '', text) # Punctuation/Numbers
    words = text.split()
    return " ".join([w for w in words if w not in STOPWORDS])


def prepare_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print(f"Reading dataset: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    # 1. UNIFY TEXT COLUMNS
    # We combine 'message' and 'email' into one 'final_text' column
    # If 'message' is empty, it takes from 'email'.
    print("Unifying text and label columns...")
    df['final_text'] = df['message'].fillna(df['email'])
    
    # 2. UNIFY LABEL COLUMNS
    # In some datasets, the label might be in a column named 'label' 
    # but in others it might be named 'v1' or something else.
    # Since your columns show 'label', we ensure it's captured.
    # If you have another label column, you can add .fillna() here too.
    df['final_label_raw'] = df['label']

    # 3. STANDARDIZE LABELS
    def unify_labels(val):
        v = str(val).lower().strip()
        if v in ['1', '1.0', 'spam', 'phishing', '1']: return 1
        if v in ['0', '0.0', 'ham', 'safe', '0']: return 0
        return None

    df['final_label'] = df['final_label_raw'].apply(unify_labels)

    # 4. DROP TRULY EMPTY DATA
    # Now we only drop rows if BOTH possible text sources were empty 
    # OR the label couldn't be figured out.
    df = df.dropna(subset=['final_text', 'final_label'])
    
    print(f"Rows surviving after unification: {len(df)}")
    
    if len(df) == 0:
        # EMERGENCY FALLBACK: If Enron has no labels, we might need to 
        # assume Enron emails are '0' (Ham/Safe) since they are corporate.
        print("Warning: No labeled rows found. Checking if Enron labels are missing...")
        # For now, let's see if the above fix works first.
        return None, None

    df['final_label'] = df['final_label'].astype(int)

    # 5. CLEAN TEXT
    print(f"Cleaning {len(df)} rows of text...")
    df['cleaned_text'] = df['final_text'].apply(clean_text)
    
    # Fallback for empty strings
    df.loc[df['cleaned_text'] == "", 'cleaned_text'] = df['final_text'].astype(str).str.lower()
    df = df[df['cleaned_text'].str.strip() != ""]

    # 6. VECTORIZATION
    vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['final_label']

    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(vectorizer, 'D:/VS Code/Python/CyberSec_Threat_Detection/models/vectorizer.pkl')
    
    return X, y


if __name__ == "__main__":
    # Test the cleaning function
    sample = "URGENT! Click here http://malicious-link.com to verify your account now!!!"
    print(f"Original: {sample}")
    print(f"Cleaned:  {clean_text(sample)}")