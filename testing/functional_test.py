import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipelines.sentiment_analysis import predict_sentiment
from pipelines.emotion_recognition import emotion_prompt, predict_emotion
from pipelines.absa import predict_absa
from warnings import filterwarnings

filterwarnings('ignore')

# ======= Sentiment Analysis =============
# Load the saved DeBERTa model and tokenizer
load_path = "D:/MSc/Sentiment & Emotion Intelligence/models/DeBERTa_model"

tokenizer = AutoTokenizer.from_pretrained(load_path)
model = AutoModelForSequenceClassification.from_pretrained(load_path)
model.eval()  # Set to evaluation mode

# ====== Emotion Recognition and ABSA ==========
load_dotenv()

# Setup API Keys
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_KEYSPACE = os.getenv('ASTRA_DB_KEYSPACE')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# HuggingFace embeddings 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Astra DB 
vector_store = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name="emotion_reviews",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE
)

# Setup LLM
llm = ChatGroq(model="llama3-70b-8192", temperature=0.3, api_key=GROQ_API_KEY)

# LLM Chain
llm_chain_emotion = LLMChain(prompt=emotion_prompt, llm=llm)
llm_chain_absa = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"))

# load labeled dataset for few shot learning
df = pd.read_csv("D:/MSc/Sentiment & Emotion Intelligence/dataset/Product Reviews - ABSA - Cleaned.csv", 
                 encoding='utf-8')

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["real_time_reviews"]
collection = db["testing"]

app = Flask(__name__)

# Sentiment Analysis - Testing
@app.route('/predict/sentiment', methods=['POST'])
def sentiment_analysis():
    data = request.get_json()
    review = data.get('review', '')
    
    if not review:
        return jsonify({'error': 'Review text is required'}), 400
    
    # Predict Sentiment Analysis
    sentiment, sentiment_conf = predict_sentiment(review, tokenizer, model)

    return jsonify({
        'sentiment': sentiment,
        'confidence': sentiment_conf
    })

# Eotion Recognition - Testing
@app.route('/predict/emotion', methods=['POST'])
def emotion_recognition():
    data = request.get_json()
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'Review text is required'}), 400

    # Predict Emotions
    emotion, emotion_conf = predict_emotion(review, vector_store, llm_chain_emotion)

    return jsonify({
        'emotion': emotion,
        'confidence': emotion_conf
    })


# ABSA - Testing
@app.route('/predict/absa', methods=['POST'])
def absa_analysis():
    data = request.get_json()
    review = data.get('review', '')
    category = data.get('category', '')

    if not review or not category:
        return jsonify({'error': 'Both review and category are required'}), 400

    # Predict Emotions
    absa = predict_absa(review, category, df, llm_chain_absa)

    return jsonify({
        'aspects': absa
    })

# Real-Time Feedback Ingestion and Storage - Testing
@app.route('/store-review', methods=['POST'])
def store_review():
    data = request.get_json()

    required_fields = ['item_id', 'product_category', 'review', 'sentiment', 'sentiment_confidence', 
                       'emotion', 'emotion_confidence', 'aspect_sentiment', 'datetime']
    
    if not all(field in data for field in required_fields):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    result = collection.insert_one(data)

    return jsonify({
        "status": "success",
        "message": "Review stored successfully.",
        "document_id": str(result.inserted_id)
    }), 200

# Main
if __name__ == '__main__':
    app.run(debug=True)