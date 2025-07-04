# fetch_results.py
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["real_time_reviews"]
collection = db["processed_feedback"]

for doc in collection.find():
    print(doc)
