# AI-Driven Sentiment & Emotion Intelligence for E-Commerce Feedback Optimization

## Overview

This project presents a real-time AI-driven solution for analyzing customer feedback in e-commerce. It utilizes advanced Natural Language Processing (NLP) techniques, a transformer-based deep learning model (DeBERTa), Large Language Models (LLMs) to extract insights related to sentiment, emotion, and product aspects from customer reviews. The insights are visualized through an interactive dashboard to support data-driven business decision-making.

### Aim

To design and implement an AI-powered feedback intelligence system capable of performing real-time sentiment analysis, emotion recognition, and aspect-based sentiment analysis (ABSA) on e-commerce customer reviews using transformer-based deep learning and retrieval-augmented generation (RAG) models.

### Objectives

- To evaluate and compare transformer-based models (BERT, RoBERTa, DeBERTa) for customer sentiment classification.
- To implement a Retrieval-Augmented Generation (RAG) approach for context-aware emotion recognition in customer feedback.
- To apply few-shot learning with large language models (LLMs) for effective ABSA on e-commerce product reviews.
- To integrate all three tasks into a real-time, scalable AI pipeline using technologies such as Apache Kafka, Spark Streaming, and MongoDB.
- To benchmark the system’s performance based on accuracy, response time, and scalability for practical deployment in business environments.

---

## Project Structure

```
.
├── pipelines                   # Prediction Pipelines
├── notebooks                   # Notebooks
├── testing                     # Functional and Non-Funtional Testing
├── real_time/
│   ├── kafka_producer.py             # Sends feedback to Kafka
│   ├── real_time_stream_processor.py # Spark Streaming to process real-time data
│   
├── feedback_analyzer.py          # Flask API for sentiment & emotion analysis
├── dashboard.py                  # Dashboard
├── app.py                        # Streamlit frontend 
├── .env                          # Environment variables (MongoDB URI, etc.)
├── requirements.txt              # Python dependencies
```

> **Database**: MongoDB (Local or Cloud)

> **Language**: Python 3.10+

> **Frameworks Required**: Apache Kafka, Apache Spark (installed locally)

---

## How to Run the Application

### Step 1: Clone the Repository

```bash
git clone https://github.com/Raeesul25/Sentiment-Emotion-Intelligence.git
cd Sentiment-Emotion-Intelligence
```

### Step 2: Set Up Environment

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Set up your `.env` file:

```bash
MONGO_URI=<your_mongodb_connection_string>
GROQ_API_KEY=<your_groq_cloud_api_key>
ASTRA_DB_API_ENDPOINT=<your_astradb_endpoint>
ASTRA_DB_APPLICATION_TOKEN=<your_astradb_token>
ASTRA_DB_KEYSPACE=<your_astradb_keyspace>
```

3. Ensure Apache Kafka and Spark are installed and running locally.

### Step 3: Download Dataset and Model

Download the dataset and trained DeBERTa model from the following OneDrive links:

- [Dataset Download Link](https://liverguac-my.sharepoint.com/:f:/g/personal/r_sally-zulfikar_rgu_ac_uk/Ep7Zy-8BCf9NmVRd75rMpBgBk1TIulvem3HhxM9ZtNDMSg?e=4x99EN)

- [Model Download Link](https://liverguac-my.sharepoint.com/:f:/g/personal/r_sally-zulfikar_rgu_ac_uk/EgJob1UWfDpImo8vMXt3IPMBNidQbzIgOIsXanK-3cseyQ?e=fCCE8A)

Extract and place them in the project root folder.

### Step 4: Run Kafka Producer

```bash
python real_time\kafka_producer.py
```

### Step 5: Run Feedback Analyzer API

```bash
python feedback_analyzer.py
```

### Step 6: Run Real-Time Stream Processor

```bash
python -m real_time.real_time_stream_processor
```

### Step 7: Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Author

Raeesul Islam – MSc Big Data Analytics – Robert Gordon University