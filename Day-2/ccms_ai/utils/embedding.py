# embedding.py

import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import hashlib

from utils.config import EMBEDDING_MODEL_NAME

# Embedding Cache

# In-memory embedding cache
embedding_cache = {}

# Generate cache key
def generate_cache_key(text: str) -> str:
    # generate a unique key for the given text input

    normalized = text.lower().strip()
    key = hashlib.sha256(normalized.encode()).hexdigest()

    return key

# Get cached embedding
def get_cached_embedding(text: str):

    key = generate_cache_key(text)

    return embedding_cache.get(key)

# Store embedding in cache
def store_embedding(text: str, embedding):

    key = generate_cache_key(text)

    embedding_cache[key] = embedding
# Embedding Model
# Global model instance
_model = None


def load_embedding_model():
    # loading the sentence transformer model only once 

    global _model

    if _model is None:
        logging.info(f"Model loaded: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model
# Generate embedding with cache support

def generate_embedding(text: str) -> np.ndarray:
    # generate embeddings for input text layer 

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Check embedding cache
    cached_embedding = get_cached_embedding(text)

    if cached_embedding is not None:
        logging.info("Embedding fetched from cache")
        return cached_embedding

    # Compute embedding
    model = load_embedding_model()

    logging.info(f"Embedding version used: {EMBEDDING_MODEL_NAME}")

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Store embedding in cache
    store_embedding(text, embedding)

    return embedding
# Combine symptoms and notes

def combine_text(symptoms: str, doctor_notes: str) -> str:
    # combining symptoms and doctor notes 

    if not isinstance(symptoms, str) or not isinstance(doctor_notes, str):
        raise TypeError("Both symptoms and doctor_notes must be strings.")

    return f"{symptoms}. {doctor_notes}"