import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "temperature": 0
}

resp = requests.post(url, headers=headers, json=payload, timeout=60)
print("status:", resp.status_code)
print(resp.text[:2000])
