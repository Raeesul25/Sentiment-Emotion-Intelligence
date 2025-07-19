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
        "review": "Purchased these for Christmas gifts. Assuming they were satisfactory because I didn't hear of need for returning. Purchased as angel tree gifts for preteens.",
        "item_id": "elect_0001"
    },
    {
        "product_category": "Tops",
        "review": "The shirt is pretty rough feeling and anywhere there is a seam, like the sleeves or the bottom of the shirt, the fabric flips upwards. Wouldn't buy again.",
        "item_id": "tops_0001"
    },
    {
        "product_category": "Amazon Alexa",
        "review": "Reviewing after one month of usage.. itâ€™s a complete delight to just order alexa to play your favorite music.. music lovers go for it..",
        "item_id": "alexa_0001"
    }
]

for review in reviews:
    producer.send('testing-topic', value=review)
    print(f"Sent: {review}")
    time.sleep(3)  # simulate real-time ingestion
