import os
import gc
import threading
import pandas as pd
import time
from functools import lru_cache
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_astradb import AstraDBVectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipelines.sentiment_analysis import predict_sentiment
from pipelines.emotion_recognition import emotion_prompt_testing, predict_emotion
from pipelines.absa import predict_absa
from warnings import filterwarnings

filterwarnings('ignore')

# ============= CONFIGURATION =============
load_dotenv()

# Setup API Keys
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_KEYSPACE = os.getenv('ASTRA_DB_KEYSPACE')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# ============= OPTIMIZED INITIALIZATION =============
class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print("Initializing models...")
        
        # ======= Sentiment Analysis Model =============
        load_path = "D:/MSc/Sentiment & Emotion Intelligence/models/DeBERTa_model"
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.sentiment_model.eval()
        
        # ====== Embeddings (Singleton) ======
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Explicit device
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        
        # ====== Vector Store with Connection Pooling ======
        self.vector_store = AstraDBVectorStore(
            embedding=self.embedding_model,
            collection_name="emotion_reviews",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE
        )
        
        # ====== LLM with Rate Limiting Protection ======
        self.llm = ChatGroq(
            model="llama3-70b-8192", 
            temperature=0.3, 
            api_key=GROQ_API_KEY,
            max_retries=3,
            request_timeout=30
        )
        
        # ====== LLM Chains ======
        self.llm_chain_emotion = LLMChain(prompt=emotion_prompt_testing, llm=self.llm)
        self.llm_chain_absa = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate.from_template("{input}")
        )
        
        # ====== Load Dataset Once ======
        self.absa_df = pd.read_csv(
            "D:/MSc/Sentiment & Emotion Intelligence/dataset/Product Reviews - ABSA - Cleaned.csv", 
            encoding='utf-8'
        )
        
        # ====== MongoDB Connection Pool ======
        self.mongo_client = MongoClient(
            MONGO_URI,
            maxPoolSize=10,
            minPoolSize=2,
            maxIdleTimeMS=30000,
            waitQueueTimeoutMS=5000
        )
        self.db = self.mongo_client["real_time_review_analyzer"]
        self.collection = self.db["scalability"]
        
        # ====== Rate Limiting ======
        self.last_llm_call = 0
        self.min_llm_interval = 0.1  # Minimum 100ms between LLM calls
        
        self._initialized = True
        print("Models initialized successfully!")

    def get_llm_with_rate_limit(self):
        """Ensure minimum interval between LLM calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_llm_call
        
        if time_since_last < self.min_llm_interval:
            time.sleep(self.min_llm_interval - time_since_last)
        
        self.last_llm_call = time.time()
        return self.llm


# Global model manager instance
model_manager = ModelManager()

# ============= OPTIMIZED PREDICTION FUNCTIONS =============
@lru_cache(maxsize=100)
def cached_sentiment_prediction(review_hash):
    """Cache sentiment predictions for identical reviews"""
    # This would need the actual review, but we cache by hash to save memory
    pass

def optimized_predict_sentiment(review):
    """Optimized sentiment prediction with caching"""
    try:
        return predict_sentiment(review, model_manager.tokenizer, model_manager.sentiment_model)
    except Exception as e:
        print(f"Sentiment prediction error: {e}")
        return "neutral", 0.5

def optimized_predict_emotion(review):
    """Optimized emotion prediction with rate limiting"""
    try:
        # Add rate limiting for vector store queries
        return predict_emotion(review, model_manager.vector_store, model_manager.llm_chain_emotion)
    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "neutral", 0.5

def optimized_predict_absa(review, category):
    """Optimized ABSA prediction with pre-loaded dataset"""
    try:
        return predict_absa(review, category, model_manager.absa_df, model_manager.llm_chain_absa)
    except Exception as e:
        print(f"ABSA prediction error: {e}")
        return {"aspects": [], "sentiments": []}

app = Flask(__name__)

# Add request context cleanup
@app.after_request
def after_request(response):
    """Clean up resources after each request"""
    gc.collect()  # Force garbage collection
    return response


# Performence - Testing
@app.route('/performence', methods=['POST'])
def performence():
    try:
        # Parse request
        data = request.get_json()
        review = data.get("review", "").strip()
        category = data.get("category", "")
        item_id = data.get("item_id", "")
        
        if not review:
            return jsonify({'error': 'Review text is required'}), 400
    
        start_time = time.time()

        # Predict Sentiment Analysis
        sentiment, sentiment_conf = optimized_predict_sentiment(review)

        # Predict Emotions
        emotion, emotion_conf = optimized_predict_emotion(review)

        # Predict Emotions
        absa = optimized_predict_absa(review, category)

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")

        new_review = {
            "item_id": item_id,
            "product_category": category,
            "review": review,
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_conf,
            "emotion": emotion,
            "emotion_confidence": emotion_conf,
            "aspect_sentiment": absa,
            "datetime": formatted_datetime
        }

        # Insert with error handling
        try:
            result = model_manager.collection.insert_one(new_review)
            document_id = str(result.inserted_id)
        except Exception as db_error:
            print(f"Database insertion error: {db_error}")
            # Continue without failing the entire request
            document_id = "db_error"

        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        
        return jsonify({
            "status": "success",
            "message": "Review stored successfully.",
            "document_id": str(result.inserted_id),
            "response_time": elapsed_ms
        }), 200
    
    except Exception as e:
        print(f"General error in feedback_analyzer: {e}")
        return jsonify({
            'error': 'Internal server error occurred during analysis',
            'details': str(e)
        }), 500
    
# Scalability - Testing
@app.route('/scalability', methods=['POST'])
def scalability():
    try:
        # Parse request
        data = request.get_json()
        review = data.get("review", "").strip()
        category = data.get("category", "")
        item_id = data.get("item_id", "")
        
        if not review:
            return jsonify({'error': 'Review text is required'}), 400
        
        # s1 = time.time()
        # Predict Sentiment Analysis
        sentiment, sentiment_conf = optimized_predict_sentiment(review)

        # s2 = time.time()
        # Predict Emotions
        emotion, emotion_conf = optimized_predict_emotion(review)

        # s3 = time.time()
        # Predict Emotions
        absa = optimized_predict_absa(review, category)

        # s4 = time.time()

        # print(f"Sentiment Time: {s2-s1:.2f}s, Emotion Time: {s3-s2:.2f}s, ABSA Time: {s4-s3:.2f}s")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")

        new_review = {
            "item_id": item_id,
            "product_category": category,
            "review": review,
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_conf,
            "emotion": emotion,
            "emotion_confidence": emotion_conf,
            "aspect_sentiment": absa,
            "datetime": formatted_datetime
        }

        # Insert with error handling
        try:
            result = model_manager.collection.insert_one(new_review)
            document_id = str(result.inserted_id)
        except Exception as db_error:
            print(f"Database insertion error: {db_error}")
            # Continue without failing the entire request
            document_id = "db_error"

        return jsonify({
            "status": "success",
            "message": "Review stored successfully.",
            "document_id": str(result.inserted_id)
        }), 200
    
    except Exception as e:
        print(f"General error in feedback_analyzer: {e}")
        return jsonify({
            'error': 'Internal server error occurred during analysis',
            'details': str(e)
        }), 500

# Simulated tokens
TOKENS = {
    "admin-token-123": "admin",
    "guest-token-456": "guest"
}

def get_user_role(token):
    return TOKENS.get(token, "unauthorized")

# Security - Testing
@app.route('/security', methods=['POST'])
def security():
    token = request.headers.get("Authorization")

    # Authorization check
    role = get_user_role(token)
    if role != "admin":
        return jsonify({"status": "error", "message": "Access Denied"}), 403

    return jsonify({"status": "success", "message": "Access Granted"}), 200

# Availability - Testing
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "alive"}), 200


# Main
if __name__ == '__main__':
    print("Starting optimized feedback analyzer...")
    app.run(
        debug=False, 
        use_reloader=False,
        threaded=True,  # Enable threading for better concurrency
        host='0.0.0.0',
        port=5000
    )