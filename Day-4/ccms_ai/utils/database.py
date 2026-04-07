# database.py

from pymongo import MongoClient
import numpy as np
from utils.config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME, EMBEDDING_MODEL_NAME

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
def fetch_all_cases():
    cases = list(collection.find({}, {"_id": 0}))

    if not cases:
        raise RuntimeError("No records found in MongoDB.")

    stored_embeddings = []
    model_mismatch_flags = []

    for case in cases:

        if "embedding" in case:
            stored_embeddings.append(case["embedding"])

            # Check embedding version mismatch
            stored_version = case.get("embedding_version", "unknown")

            if stored_version != EMBEDDING_MODEL_NAME:
                model_mismatch_flags.append(True)
            else:
                model_mismatch_flags.append(False)
    stored_embeddings = np.array(stored_embeddings, dtype=np.float32)

    return cases, stored_embeddings, model_mismatch_flags