# Few-Shot Prompting Practice
# Based on Week 4 resource: few-shot prompting examples.
# Few-shot prompting means giving example input-output pairs.

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Convert long questions into short and clear questions."
        },
        {
            "role": "user",
            "content": "I have three apples and my brother has four apples. How many apples do we have in total?"
        },
        {
            "role": "assistant",
            "content": "What is 3 + 4?"
        },
        {
            "role": "user",
            "content": "Can you explain to me what Python programming language is and why people use it?"
        }
    ],
    temperature=0.3
)

answer = response.choices[0].message.content
print(answer)