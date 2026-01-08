import os

import azure.identity
import openai
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST", "github")
client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
messages = [
    {"role": "system", "content": "I am a large language model."},
]

while True:
    question = input("\nYour question: ")
    print("Sending question...")

    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        stream=True,
    )

    print("\nAnswer: ")
    bot_response = ""
    for event in response:
        if event.choices and event.choices[0].delta.content:
            content = event.choices[0].delta.content
            print(content, end="", flush=True)
            bot_response += content
    print("\n")
    messages.append({"role": "assistant", "content": bot_response})