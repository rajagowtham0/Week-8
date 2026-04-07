# app.py

from fastapi import FastAPI, HTTPException
import time
import logging
import psutil
import os

from utils.models import CaseRequest, CaseResponse
from utils.embedding import combine_text
from utils.config import EMBEDDING_MODEL_NAME
# Retrieval Engine
from retrieval.retrieval_engine import initialize_engine, analyze_case

logging.basicConfig(level=logging.INFO)

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


app = FastAPI(title="CCMS AI Similarity Engine")

# API response cache
cache = {}

# ✅ NEW: dataset size tracker
DATASET_SIZE = 0


# Startup Event
@app.on_event("startup")
def load_data():

    global DATASET_SIZE

    logging.info(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    try:
        initialize_engine()
        logging.info("Retrieval engine initialized successfully.")

        # ✅ Get dataset size (number of indexed cases)
        from retrieval.retrieval_engine import case_ids
        DATASET_SIZE = len(case_ids)

        logging.info(f"[PERF] Dataset size: {DATASET_SIZE}")

    except Exception as e:
        logging.warning(f"Retrieval engine initialization failed: {e}")


# Main API Endpoint
@app.post("/analyze-case", response_model=CaseResponse)
def analyze_case_api(request: CaseRequest):

    global cache, DATASET_SIZE

    api_start_time = time.time()

    try:
        logging.info("Received API request")

        # Input validation
        if not request.symptoms.strip() or not request.doctor_notes.strip():
            raise HTTPException(
                status_code=400,
                detail="Symptoms and doctor notes cannot be empty."
            )

        # Combine input
        combined_text = combine_text(
            request.symptoms,
            request.doctor_notes
        )

        logging.info("Input combined successfully")

        cache_key = combined_text.lower().strip()

        # Cache check
        if cache_key in cache:
            logging.info("API response fetched from cache")

            api_response_time = round(time.time() - api_start_time, 4)

            logging.info(f"[PERF] Dataset Size: {DATASET_SIZE}")
            logging.info(f"[PERF] Retrieval Time: 0.0 sec (cache)")
            logging.info(f"[PERF] API Total Response Time: {api_response_time} sec")

            return cache[cache_key]

        # Measure Retrieval Time separately
        retrieval_start_time = time.time()

        logging.info("Calling retrieval engine")
        insight_output = analyze_case(combined_text)

        retrieval_time = round(time.time() - retrieval_start_time, 4)

        logging.info("Insight generated successfully")

        # Store in cache
        cache[cache_key] = insight_output

        # Total API time
        api_response_time = round(time.time() - api_start_time, 4)

        # Memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)

        # PERFORMANCE LOGS 
        logging.info(f"[PERF] Dataset Size: {DATASET_SIZE}")
        logging.info(f"[PERF] Retrieval Time: {retrieval_time} sec")
        logging.info(f"[PERF] API Total Response Time: {api_response_time} sec")
        logging.info(f"[PERF] Memory Usage: {round(memory_usage,2)} MB")

        return insight_output

    except Exception as e:

        logging.error(f"Error occurred: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )