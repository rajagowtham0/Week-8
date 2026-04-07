# retrieval_engine.py

import numpy as np
import re
import logging
from pymongo import MongoClient
from collections import Counter
from retrieval.vector_index import VectorIndex
from utils.embedding import generate_embedding
from utils.config import (
    MONGO_URI,
    DATABASE_NAME,
    COLLECTION_NAME,
    TOP_N
)

vector_index = None
case_ids = []
stored_cases = []
engine_initialized = False


def initialize_engine():

    global vector_index, case_ids, stored_cases, engine_initialized

    if engine_initialized:
        logging.info("Retrieval engine already initialized.")
        return

    logging.info("Initializing retrieval engine...")

    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    docs = list(collection.find())

    if not docs:
        raise RuntimeError("No embeddings found in MongoDB.")

    embeddings = []
    case_ids = []
    stored_cases = []

    for doc in docs:
        if "embedding" in doc and "case_id" in doc:
            embeddings.append(doc["embedding"])
            case_ids.append(doc["case_id"])
            stored_cases.append(doc)

    vector_index = VectorIndex()
    vector_index.build_index(embeddings, case_ids)

    logging.info("Vector index built successfully")
    logging.info(f"Total vectors indexed: {len(case_ids)}")

    engine_initialized = True


def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def retrieve_similar_cases(text, top_k=TOP_N):

    global vector_index

    if not engine_initialized:
        raise RuntimeError("Call initialize_engine() first.")

    text = preprocess_text(text)

    if "." in text:
        parts = text.split(".", 1)
        symptoms = parts[0].strip()
        doctor_notes = parts[1].strip()
    else:
        symptoms = text
        doctor_notes = ""

    embedding = generate_embedding(symptoms, doctor_notes)

    results = vector_index.search(embedding, top_k)

    logging.info("FAISS top-K search executed")

    return results


def extract_shared_symptoms(query_text, case_text):

    stopwords = {"and", "with", "the", "on", "in", "of", "to"}

    query_words = set(query_text.lower().split()) - stopwords
    case_words = set(case_text.lower().split()) - stopwords

    overlap = query_words.intersection(case_words)

    return list(overlap)[:3]


def generate_case_insight(similar_cases, query_text):

    global stored_cases

    if not similar_cases:
        return {
            "similar_cases": [],
            "symptoms": "",
            "treatment": "No treatment pattern identified.",
            "similarity_score": "0.0"
        }

    filtered_cases = [
        case for case in similar_cases
        if case["similarity_score"] >= 0.5
    ]

    if not filtered_cases:
        return {
            "similar_cases": [],
            "symptoms": "",
            "treatment": "No reliable similar cases identified.",
            "similarity_score": "0.0"
        }

    filtered_cases.sort(key=lambda x: x["similarity_score"], reverse=True)
    filtered_cases = filtered_cases[:4]

    case_map = {case["case_id"]: case for case in stored_cases}

    similarity_scores = []
    treatments = []
    structured_cases = []

    for case in filtered_cases:

        case_id = case["case_id"]
        score = float(case["similarity_score"])

        similarity_scores.append(score)

        structured_cases.append({
            "case_id": case_id
        })

        matched_case = case_map.get(case_id)

        if matched_case and "treatment" in matched_case:
            treatments.append(matched_case["treatment"])

    if treatments:
        treatment_raw = Counter(treatments).most_common(1)[0][0]
    else:
        treatment_raw = "No treatment pattern identified."

    mean_similarity = round(
        sum(similarity_scores) / len(similarity_scores),
        4
    )

    symptoms_output = []

    for case in filtered_cases:
        case_id = case["case_id"]
        matched_case = case_map.get(case_id)

        if matched_case and "symptoms" in matched_case:
            overlap = extract_shared_symptoms(
                query_text,
                matched_case["symptoms"]
            )
            if overlap:
                symptoms_output = overlap
                break

    if not symptoms_output and filtered_cases:
        top_case = case_map.get(filtered_cases[0]["case_id"])
        if top_case and "symptoms" in top_case:
            symptoms_output = top_case["symptoms"].split()[:3]

    # FINAL FORMATTED OUTPUT
    return {
        "similar_cases": structured_cases,
        "symptoms": f"The similarity is mainly due to shared symptoms such as {', '.join(symptoms_output)}",
        "treatment": f"In similar past cases, patients well responded {treatment_raw}",
        "similarity_score": f"Based on the {len(filtered_cases)} similar patients, the weighted confidence score obtained is {mean_similarity}"
    }


def analyze_case(text):

    similar_cases = retrieve_similar_cases(text)

    insight = generate_case_insight(similar_cases, text)

    return insight