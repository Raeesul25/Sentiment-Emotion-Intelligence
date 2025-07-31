import json
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

reviews = [
    {
        "product_category": "Electronics",
        "review": "There is a problem with mic.its not loud nd clear.else works fine , build is delicate.",
        "item_id": "elect_0001"
    },
    {
        "product_category": "Electronics",
        "review": "Please don't buy anyone , after 10 days it not work properly ,  power button is not working  always it gives maximum volume alert",
        "item_id": "elect_0001"
    },
    {
        "product_category": "Electronics",
        "review": "Not good product sound quality average bass is good I recommend to sennsizer CX 180 For Bass Lover n sound quality",
        "item_id": "elect_0001"
    }
]

for review in reviews:
    producer.send('feedback-topic', value=review)
    print(f"Sent: {review}")
    time.sleep(3)  # simulate real-time ingestion
