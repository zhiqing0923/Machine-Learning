import os

import azure.identity
import openai
from dotenv import load_dotenv

load_dotenv(override=True)

API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.7,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that makes lots of cat references and uses emojis."},
        {"role": "user", "content": "Write about a hungry cat who wants tuna"},
    ],
    stream=True,
)

print(f"Response from {API_HOST}: \n")
# print(response.choices[0].message.content)
for event in response:
    if event.choices:
        content = event.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

