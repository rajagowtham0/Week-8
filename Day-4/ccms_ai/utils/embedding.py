# embedding.py

import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import hashlib
from pymongo import MongoClient

from utils.config import (
    EMBEDDING_MODEL_NAME,
    MONGO_URI,
    DATABASE_NAME,
    COLLECTION_NAME
)

# MongoDB Connection
client = MongoClient(MONGO_URI)
collection = client[DATABASE_NAME][COLLECTION_NAME]

# Embedding Cache
embedding_cache = {}

def generate_cache_key(text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

def get_cached_embedding(text: str):
    key = generate_cache_key(text)
    return embedding_cache.get(key)

def store_embedding(text: str, embedding):
    key = generate_cache_key(text)
    embedding_cache[key] = embedding

# Model Loading
_model = None

def load_embedding_model():
    global _model
    if _model is None:
        logging.info(f"Model loaded: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

# Generate Embedding 
def generate_embedding(symptoms: str, doctor_notes: str) -> np.ndarray:

    if not isinstance(symptoms, str) or not isinstance(doctor_notes, str):
        raise TypeError("Inputs must be strings.")

    combined_text = f"{symptoms}. {doctor_notes}"

    # 1. Check cache
    cached = get_cached_embedding(combined_text)
    if cached is not None:
        logging.info("Embedding fetched from cache")
        return cached

    # 2. Check MongoDB
    doc = collection.find_one({
        "symptoms": symptoms,
        "doctor_notes": doctor_notes
    })

    if doc and "embedding" in doc:
        logging.info("Embedding fetched from MongoDB")

        embedding = np.array(doc["embedding"], dtype=np.float32)
        store_embedding(combined_text, embedding)
        return embedding

    # 3. Generate embedding
    model = load_embedding_model()

    logging.info(f"Generating embedding using: {EMBEDDING_MODEL_NAME}")

    embedding = model.encode(
        combined_text,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # 4. Store in MongoDB 
    collection.update_one(
        {
            "symptoms": symptoms,
            "doctor_notes": doctor_notes
        },
        {
            "$set": {
                "symptoms": symptoms,
                "doctor_notes": doctor_notes,
                "embedding": embedding.tolist(),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_version": EMBEDDING_MODEL_NAME
            }
        },
        upsert=True
    )

    # 5. Store in cache
    store_embedding(combined_text, embedding)

    return embedding

# Combine Text
def combine_text(symptoms: str, doctor_notes: str) -> str:

    if not isinstance(symptoms, str) or not isinstance(doctor_notes, str):
        raise TypeError("Both inputs must be strings.")

    return f"{symptoms}. {doctor_notes}"