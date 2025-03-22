import requests
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('sentence', type=str)

args = parser.parse_args()

url = "http://127.0.0.1:8000/predict"
data = {"input": args.sentence}

print(f'Sending\n{data}')

try:
    response = requests.post(url, json=data)
    response.raise_for_status()
    prediction = response.json()
    print(prediction)

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")