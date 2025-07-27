import findspark
findspark.init()

import os
import logging
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType

os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';C:\\hadoop\\bin'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('RealTimeStreaming')

# === Spark Session ===
spark = SparkSession \
    .builder \
    .appName("Streaming from Kafka") \
    .master("local[*]") \
    .config("spark.streaming.stopGracefullyOnShutdown", True) \
    .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6') \
    .config("spark.sql.shuffle.partitions", 4) \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configurationFile=log4j2.properties") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate() 

spark.sparkContext.setLogLevel("ERROR")

# === Kafka Stream Schema ===
schema = StructType() \
    .add("product_category", StringType()) \
    .add("review", StringType()) \
    .add("item_id", StringType())

# === Streaming Input from Kafka ===
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "testing-topic") \
    .option("startingOffsets", "earliest") \
    .load()

json_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# === Process each row ===
def process_batch(df, epoch_id):
    rows = df.collect()
    for row in rows:
        review = row['review']
        category = row['product_category']
        item_id = row['item_id']
        
        payload = {
            "item_id": item_id,
            "review": review,
            "category": category
        }
        
        try:
            logger.info(f"Received a new review to analysis...")
            response = requests.post("http://localhost:5000/analyze", json=payload)
            result = response.json()
            logger.info(f"Review analyzed: {result}")
        
        except Exception as e:
            logger.error(f"Error processing row: {e}")


# === Write to MongoDB ===
query = json_df.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()
