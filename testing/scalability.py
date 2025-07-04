import requests
import json
import time
import pandas as pd
from tqdm import tqdm

# function for time format
def ms_to_time_format(ms):
    hours = int(ms // (1000 * 60 * 60))
    minutes = int((ms % (1000 * 60 * 60)) // (1000 * 60))
    seconds = int((ms % (1000 * 60)) // 1000)
    milliseconds = int(round(ms % 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

df = pd.read_csv("D:/MSc/Sentiment & Emotion Intelligence/dataset/Scalability Testing.csv")

total_requests = df.shape[0]
total_time = 0
pass_count = 0

for i in tqdm(range(total_requests), desc="Processing Reviews"):
    review = {
        "item_id": df.iloc[i, 0],
        "category": df.iloc[i, 1],
        "review": df.iloc[i, 2]
    }

    start = time.time()
    res = requests.post('http://localhost:5000/scalability', json=review)
    end = time.time()
    response_time = (end - start) * 1000  # ms

    total_time += response_time

    if response_time <= 5000:
        pass_count += 1

    time.sleep(5)

print(f"Total Requests: {total_requests}")
print(f"Total Response Time for {total_requests} Requests: {ms_to_time_format(total_time)}")
print(f"Average Response Time: {total_time / total_requests:.2f} ms")
print(f"Pass Rate: {(pass_count / total_requests) * 100:.2f}%")
