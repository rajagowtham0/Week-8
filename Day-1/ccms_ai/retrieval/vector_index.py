# vector_index.py

import numpy as np
import faiss
import logging


class VectorIndex:

    def __init__(self):
        self.index = None
        self.case_ids = []
    # Build FAISS index
    def build_index(self, embeddings, case_ids):

        if not embeddings or not case_ids:
            raise ValueError("Embeddings or case_ids are empty")

        if len(embeddings) != len(case_ids):
            raise ValueError("Embeddings and case_ids length mismatch")

        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.case_ids = case_ids

        logging.info(f"FAISS index built with {self.index.ntotal} vectors")

    # Search similar cases
    def search(self, query_embedding, top_k=5):

        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):

            case_id = self.case_ids[idx]

            score = 1 / (1 + distances[0][rank])

            results.append({
                "case_id": case_id,
                "similarity_score": round(float(score), 4)
            })

        return results