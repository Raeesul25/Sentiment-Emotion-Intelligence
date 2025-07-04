import subprocess
import time

while True:
    print("Starting Flask server...")
    process = subprocess.Popen(["python", "non_functional_test.py"])
    process.wait()  # Wait until it crashes or is stopped
    print("Server crashed! Restarting in 5 seconds...")
    time.sleep(5)
