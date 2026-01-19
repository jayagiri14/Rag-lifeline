"""Simple embeddings using TF-IDF as fallback - no external dependencies."""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize the vectorizer
_vectorizer = None
_fitted = False

def get_vectorizer():
    """Get or create the TF-IDF vectorizer."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(max_features=768, stop_words='english')
    return _vectorizer

def fit_vectorizer(texts: list[str]):
    """Fit the vectorizer on texts."""
    global _fitted
    vectorizer = get_vectorizer()
    vectorizer.fit(texts)
    _fitted = True

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using TF-IDF."""
    global _fitted
    vectorizer = get_vectorizer()
    
    # Fit on first call if not fitted
    if not _fitted:
        vectorizer.fit(texts)
        _fitted = True
    
    # Transform texts
    tfidf_matrix = vectorizer.transform(texts)
    
    # Pad or truncate to exact 768 dimensions
    result = np.zeros((len(texts), 768))
    actual_features = min(tfidf_matrix.shape[1], 768)
    result[:, :actual_features] = tfidf_matrix.toarray()[:, :actual_features]
    
    # L2 normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    result = result / norms
    
    return result.tolist()

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return get_embeddings([text])[0]
