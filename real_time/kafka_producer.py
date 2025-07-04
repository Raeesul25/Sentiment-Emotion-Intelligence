import json
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

reviews = [
    {"product_category": "Electronics",
     "review": "I got this tablet on a deal and has good quality of video you can watch.",
     "item_id": "elect_0001"},
]

for review in reviews:
    producer.send('testing-topic', value=review)
    print(f"Sent: {review}")
    time.sleep(3)  # simulate real-time ingestion
