# OpenAI API Basics Practice
# Based on Week 4 resource: GPT API basics.
# Do not write API keys directly in code.

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain what a large language model is in simple words."
        }
    ],
    temperature=0.7
)

answer = response.choices[0].message.content
print(answer)